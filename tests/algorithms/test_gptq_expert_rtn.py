"""E8 expert RTN path unit tests (gptq_algo.py expert_mode='rtn').

Covers: mode validation, GPTQModule RTN numeric correctness (half-step bound,
grid membership), Hessian-skip behavior, BatchedGPTQModule RTN + write-back,
and pack-path compatibility with GPTQ packing (same scales/zeros/g_idx layout).
"""

import math

import pytest
import torch

from npuslim.algorithms.quantization.gptq.gptq_algo import (
    BatchedGPTQModule,
    GPTQAlgorithm,
    GPTQModule,
    _ExpertSliceLinear,
)
from npuslim.algorithms.quantization.hessian.hessian_common import (
    compute_scales_with_zero,
    quantize_with_scale_zero,
)


def test_expert_mode_validation():
    with pytest.raises(NotImplementedError):
        GPTQAlgorithm(wbits=4, group_size=128, expert_mode="ssz")
    with pytest.raises(ValueError):
        GPTQAlgorithm(wbits=4, group_size=128, expert_mode="bogus")
    algo = GPTQAlgorithm(wbits=4, group_size=128, expert_mode="RTN")
    assert algo.expert_mode == "rtn"
    assert algo._expert_rtn is True
    algo_default = GPTQAlgorithm(wbits=4, group_size=128)
    assert algo_default.expert_mode is None
    assert algo_default._expert_rtn is False


def _make_module(out=4, inp=8, groupsize=4, rtn=True):
    torch.manual_seed(7)
    linear = torch.nn.Linear(inp, out, bias=False)
    return GPTQModule(
        linear,
        wbits=4,
        groupsize=groupsize,
        sym=True,
        percdamp=0.01,
        preproc_hessian=True,
        rtn=rtn,
    )


def test_gptq_module_rtn_quantize_and_hessian_skip():
    mod = _make_module()
    # Hessian collection in rtn mode: skipped, activation stats kept
    x = torch.randn(3, 8)
    mod.add_batch(x, None)
    mod.add_batch(x, None)
    assert mod.H is None
    assert mod.nsamples == 6
    assert mod._x_abs_sum is not None and mod._x_abs_sum.shape == (8,)
    mod.preproc()
    assert mod.preproc_done

    scales, zeros, g_idx = mod.fasterquant(layer_name="test.experts.0.gate_up_proj")
    assert scales.shape == (4, 2)
    assert zeros.shape == (4, 2)
    assert g_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    m = mod.last_metrics
    assert m["rows"] == 4 and m["columns"] == 8
    assert m["fallback"] is None
    assert m["norm_loss"] >= 0.0 and math.isfinite(m["avg_loss"])


def test_gptq_module_rtn_half_step_bound_and_grid():
    mod = _make_module()
    w0 = mod.layer.weight.data.clone()
    mod.preproc()
    scales, zeros, _ = mod.fasterquant()
    q = mod.layer.weight.data.float()
    # grid membership: every quantized value lies on some group's grid
    for g in range(2):
        w_blk = w0[:, g * 4 : (g + 1) * 4].float()
        qs = q[:, g * 4 : (g + 1) * 4]
        s = scales[:, g]
        # symmetric 4-bit: q = s * clamp(round(w/s) + 8, 0, 15) - s * 8
        k_pre = torch.round(w_blk / s.unsqueeze(1))
        k = torch.round(qs / s.unsqueeze(1))
        assert torch.allclose(qs, s.unsqueeze(1) * k, atol=1e-5)
        assert (k >= -8).all() and (k <= 7).all()  # 4-bit symmetric grid range
        # RTN half-step property holds wherever the round is not clamped
        nonclamped = (k_pre >= -8) & (k_pre <= 7)
        res = (qs - w_blk).abs()
        assert (res[nonclamped] <= s.unsqueeze(1).expand_as(res)[nonclamped] / 2 + 1e-4).all()


def test_batched_rtn_module_write_back():
    torch.manual_seed(3)
    CH = 3
    weight_3d = torch.nn.Parameter(torch.randn(CH, 4, 8))
    slices = [_ExpertSliceLinear(weight_3d, e, "gate_up_proj", "mlp.experts") for e in range(CH)]
    mod = BatchedGPTQModule(
        slices, wbits=4, groupsize=4, sym=True, percdamp=0.01,
        preproc_hessian=True, hessian_device=torch.device("cpu"), rtn=True,
    )
    assert mod.H is None
    x = torch.randn(5, 8)
    for e in range(CH):
        mod.add_batch_for_expert(x, e)
    assert mod.H is None  # skipped
    assert mod.nsamples.tolist() == [5, 5, 5]
    mod.preproc()
    w0_pre = [weight_3d.data[e].clone() for e in range(CH)]  # before quantization
    all_scales, all_zeros, g_idx = mod.fasterquant(layer_name="test.experts.gate_up_proj")
    mod.write_back()
    assert all_scales.shape == (CH, 4, 2)
    assert all_zeros.shape == (CH, 4, 2)
    assert g_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert mod.last_metrics["fallback"] is None
    for e in range(CH):
        w0 = w0_pre[e]
        wq = slices[e].weight.data
        assert wq.shape == (4, 8)
        # values changed (quantized) but write_back synced slices into the 3D param
        assert not torch.equal(wq.float(), w0.float())
        assert torch.equal(weight_3d.data[e].float(), wq.float())


def test_rtn_pack_path_compatible():
    """RTN outputs must drive the same packing path as GPTQ (E8 requirement)."""
    algo = GPTQAlgorithm(wbits=4, group_size=4, expert_mode="rtn")
    mod = _make_module()
    mod.preproc()
    scales, zeros, g_idx = mod.fasterquant()
    packed, names = algo._pack_quant_linear_tensors(
        module_name="mlp.experts.0.gate_up_proj",
        linear_module=mod.layer,
        scales=scales,
        zeros=zeros,
        g_idx=g_idx,
    )
    assert set(packed.keys()) == {
        "mlp.experts.0.gate_up_proj.qweight",
        "mlp.experts.0.gate_up_proj.qzeros",
        "mlp.experts.0.gate_up_proj.scales",
        "mlp.experts.0.gate_up_proj.g_idx",
    }
    assert packed["mlp.experts.0.gate_up_proj.g_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_rtn_matches_reference_rtn():
    """RTN quantization equals manual scale->round->dequant on the same grid."""
    mod = _make_module()
    w0 = mod.layer.weight.data.float()
    scales_ref = []
    for g in range(2):
        s, z = compute_scales_with_zero(w0[:, g * 4 : (g + 1) * 4], bits=4, sym=True)
        scales_ref.append((s, z))
    mod.preproc()
    mod.fasterquant()
    q = mod.layer.weight.data.float()
    for g, (s, z) in enumerate(scales_ref):
        ref = quantize_with_scale_zero(
            w0[:, g * 4 : (g + 1) * 4], s, z, bits=4
        )
        assert torch.allclose(q[:, g * 4 : (g + 1) * 4], ref, atol=1e-6)
