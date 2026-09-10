"""SmoothQuant (norm-linear) unit tests for GPTQAlgorithm.

Covers: config wiring, default GLM norm-target scope, smooth_scale fold math
(fold keeps the subgraph function identical), norm snapshot sync, and the
multi-target combined weight scale semantics (modelslim compute_multi_weight_scale).
"""

import pytest
import torch
import torch.nn as nn

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQAlgorithm
from npuslim.algorithms.quantization.scale_search import smooth_scale


class _Norm(nn.Module):
    """Tiny RMSNorm-like gain module for testing (y = gamma * x)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x):
        return x * self.weight


class _FakeSmoothLayer(nn.Module):
    """Mimics a GLM layer: one norm + two consumer linears."""

    def __init__(self, hidden=8):
        super().__init__()
        self.name = "test.0"
        self.input_layernorm = _Norm(hidden)
        self.self_attn = nn.Module()
        self.self_attn.q_a_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.kv_a_proj_with_mqa = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.q_a_layernorm = _Norm(6)
        self.self_attn.q_b_proj = nn.Linear(6, hidden, bias=False)

    def forward(self, x):
        h = self.input_layernorm(x)
        qa = self.self_attn.q_a_proj(h)
        kva = self.self_attn.kv_a_proj_with_mqa(h)
        qb = self.self_attn.q_b_proj(self.self_attn.q_a_layernorm(torch.randn_like(qa[..., :6])))
        return qa + kva + qb


def test_smooth_quant_config_wiring():
    algo = GPTQAlgorithm(wbits=4, group_size=-1, smooth_quant={"alpha": 0.5})
    assert algo.smooth_quant_cfg == {"alpha": 0.5}
    assert algo._smooth_norm_targets() == GPTQAlgorithm.DEFAULT_SMOOTH_NORM_TARGETS
    algo_default = GPTQAlgorithm(wbits=4, group_size=-1)
    assert algo_default.smooth_quant_cfg is None
    assert algo_default._smooth_norm_targets() == []


def test_smooth_scale_math():
    torch.manual_seed(3)
    x_max = torch.rand(8) * 5 + 0.5
    w_max = torch.rand(8) * 5 + 0.5
    s = smooth_scale(x_max, w_max, alpha=0.5)
    # s = |X|^a / |W|^(1-a); for a=0.5: s^2 = X/W
    assert torch.allclose(s.pow(2), x_max / w_max, atol=1e-5)
    assert (s > 0).all()


def test_smooth_quant_fold_keeps_function():
    """Folding γ/s + W*s must keep the norm->linear output identical."""
    import torch.nn as nn

    torch.manual_seed(5)
    layer = _FakeSmoothLayer(hidden=8)
    algo = GPTQAlgorithm(
        wbits=4,
        group_size=-1,
        smooth_quant={"alpha": 0.5, "targets": [{"norm": "input_layernorm",
                                                 "targets": ["self_attn.q_a_proj"]}]},
    )
    gamma0 = layer.input_layernorm.weight.detach().clone()
    w0 = layer.self_attn.q_a_proj.weight.detach().clone()
    x = torch.randn(3, 4, 8)

    # before fold
    y0 = layer.self_attn.q_a_proj(x * layer.input_layernorm.weight)

    # collect stats like _collect_statistics would (abs-max of norm output)
    acc = [None]
    layer.input_layernorm.register_forward_hook(
        lambda m, i, o, acc=acc: algo._smooth_collect(o, acc)
    )
    layer(x)
    algo._smooth_norm_stats["test.0.input_layernorm"] = acc
    assert acc[0] is not None and acc[0].shape == (8,)

    # apply fold
    algo._apply_smooth_quant(layer, chunk=None)
    assert not torch.equal(layer.input_layernorm.weight.detach(), gamma0)  # γ changed
    assert not torch.equal(layer.self_attn.q_a_proj.weight.detach(), w0)  # W changed

    # after fold: same function
    y1 = layer.self_attn.q_a_proj(x * layer.input_layernorm.weight)
    assert torch.allclose(y0, y1, atol=1e-5)


def test_smooth_quant_multi_target_combined_weight_scale():
    """Per-input-channel w_scale = max across all targets of the same norm."""
    import torch.nn as nn

    torch.manual_seed(9)
    layer = _FakeSmoothLayer(hidden=8)
    algo = GPTQAlgorithm(
        wbits=4,
        group_size=-1,
        smooth_quant={"alpha": 0.5, "targets": [{"norm": "input_layernorm",
                                                 "targets": ["self_attn.q_a_proj",
                                                             "self_attn.kv_a_proj_with_mqa"]}]},
    )
    acc = [None]
    layer.input_layernorm.register_forward_hook(
        lambda m, i, o, acc=acc: algo._smooth_collect(o, acc)
    )
    layer(torch.randn(2, 3, 8))
    algo._smooth_norm_stats["test.0.input_layernorm"] = acc

    w_a = layer.self_attn.q_a_proj.weight.detach().clone().float()
    w_k = layer.self_attn.kv_a_proj_with_mqa.weight.detach().clone().float()
    w_max_ref = torch.stack([
        w_a.abs().amax(dim=0), w_k.abs().amax(dim=0)
    ]).amax(dim=0)
    algo._apply_smooth_quant(layer, chunk=None)
    # both targets folded by the same scale; norm folded by 1/s
    s = smooth_scale(acc[0], w_max_ref, alpha=0.5)
    assert torch.allclose(
        layer.self_attn.q_a_proj.weight.detach().float(),
        w_a * s.view(1, -1),
        atol=1e-5,
    )
    assert torch.allclose(
        layer.self_attn.kv_a_proj_with_mqa.weight.detach().float(),
        w_k * s.view(1, -1),
        atol=1e-5,
    )
    assert torch.allclose(
        layer.input_layernorm.weight.detach().float(),
        torch.ones(8) / s,
        atol=1e-5,
    )


def test_smooth_quant_requires_stats():
    """Without collected stats the fold is a no-op (no crash)."""
    import torch.nn as nn

    layer = _FakeSmoothLayer(hidden=8)
    algo = GPTQAlgorithm(wbits=4, group_size=-1, smooth_quant={"alpha": 0.5})
    gamma0 = layer.input_layernorm.weight.detach().clone()
    algo._apply_smooth_quant(layer, chunk=None)
    assert torch.equal(layer.input_layernorm.weight.detach(), gamma0)
