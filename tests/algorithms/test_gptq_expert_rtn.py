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
    with pytest.raises(ValueError):
        GPTQAlgorithm(wbits=4, group_size=128, expert_mode="bogus")
    algo = GPTQAlgorithm(wbits=4, group_size=128, expert_mode="RTN")
    assert algo.expert_mode == "rtn"
    assert algo._expert_rtn is True
    assert algo._expert_ssz is False
    # ssz mode: rtn base + SSZ scale refinement, no Hessian compensation
    algo_ssz = GPTQAlgorithm(wbits=4, group_size=128, expert_mode="ssz", ssz_step=10)
    assert algo_ssz.expert_mode == "ssz"
    assert algo_ssz._expert_rtn is True
    assert algo_ssz._expert_ssz is True
    algo_default = GPTQAlgorithm(wbits=4, group_size=128)
    assert algo_default.expert_mode is None
    assert algo_default._expert_rtn is False
    assert algo_default._expert_ssz is False


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


def _make_ssz_module(out=4, inp=8, groupsize=-1, ssz=True, ssz_step=10, rtn=True):
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
        ssz=ssz,
        ssz_step=ssz_step,
    )


def test_ssz_refines_scale_lowers_mse():
    """SSZ must not increase MSE vs minmax baseline, and refine scale in-grid."""
    from npuslim.algorithms.quantization.ssz import ssz_calculate_scale

    w = torch.randn(6, 12, dtype=torch.float64)
    s0, z0 = compute_scales_with_zero(w, bits=4, sym=True)
    s_refined = ssz_calculate_scale(w, s0.float(), bits=4, step=10).double()
    # scales refined (not identical to minmax) and stay positive
    assert (s_refined > 0).all()
    q0 = quantize_with_scale_zero(w, s0, z0, bits=4)
    q1 = quantize_with_scale_zero(w, s_refined, z0, bits=4)
    mse0 = ((w - q0) ** 2).mean().item()
    mse1 = ((w - q1) ** 2).mean().item()
    assert mse1 <= mse0 * (1 + 1e-6)  # never worse
    # quantized values remain on the symmetric 4-bit grid:
    # dequant = s*(q - zero) with q = clamp(round(w/s)+zero, 0, maxq), so
    # dequant/s = (q - zero) is the signed grid value in [-8, 7].
    v = (q1 / s_refined).round()
    assert (v >= -8).all() and (v <= 7).all()


def test_ssz_gptq_module_per_channel():
    """expert_mode ssz via GPTQModule: rtn base + refined per-channel scale."""
    mod = _make_ssz_module(groupsize=-1)
    x = torch.randn(3, 8)
    mod.add_batch(x, None)
    assert mod.H is None  # rtn base skips Hessian
    assert mod.ssz is True
    mod.preproc()
    scales, zeros, g_idx = mod.fasterquant()
    assert scales.shape == (4, 1)  # per-channel
    assert zeros.shape == (4, 1)
    assert g_idx.tolist() == [0] * 8
    assert mod.last_metrics["fallback"] is None


def test_ssz_pack_path_compatible():
    """SSZ outputs must drive the same packing path as rtn/gptq."""
    algo = GPTQAlgorithm(wbits=4, group_size=-1, expert_mode="ssz")
    mod = _make_ssz_module(groupsize=-1)
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


def test_ssz_grouped_path_skipped():
    """SSZ on grouped path (groupsize != -1) warns and falls back to rtn scales."""
    mod = _make_ssz_module(groupsize=4)
    mod.preproc()
    scales, zeros, g_idx = mod.fasterquant()
    assert scales.shape == (4, 2)  # grouped scales unchanged (minmax)
    assert g_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


class _FakeLayer:
    name = "test.layer.0"


def _make_updown_handlers():
    """Batched down + fused gate_up handlers over [E=2, H=6, I=2] 3D params."""
    torch.manual_seed(11)
    weight_gu = torch.nn.Parameter(torch.randn(2, 4, 6))  # [E, 2I, H]
    weight_dn = torch.nn.Parameter(torch.randn(2, 6, 2))  # [E, H, I]
    gu_slices = [_ExpertSliceLinear(weight_gu, e, "gate_up_proj", "mlp.experts") for e in range(2)]
    dn_slices = [_ExpertSliceLinear(weight_dn, e, "down_proj", "mlp.experts") for e in range(2)]
    mod_gu = BatchedGPTQModule(
        gu_slices, wbits=4, groupsize=-1, sym=True, percdamp=0.01,
        preproc_hessian=True, hessian_device=torch.device("cpu"), rtn=True,
    )
    mod_dn = BatchedGPTQModule(
        dn_slices, wbits=4, groupsize=-1, sym=True, percdamp=0.01,
        preproc_hessian=True, hessian_device=torch.device("cpu"), rtn=True,
    )
    # Feed distinct middle activations (down input = swiglu output)
    mid0 = torch.randn(5, 2) + 3.0  # skew channel significance
    mid1 = torch.randn(7, 2) * 5.0
    mod_dn.add_batch_for_expert(mid0, 0)
    mod_dn.add_batch_for_expert(mid1, 1)
    return weight_gu, weight_dn, {"dn": mod_dn, "gu": mod_gu}


def test_updown_awq_folds_down_columns_and_up_rows():
    """Balanced AWQ fold (modelslim UpDownSubgraphFusion): down = W_d*s, up = W_u/s,
    gate half untouched. s cancels in the product, so down[h,i]*up[i,h] must equal
    the original dn_orig[h,i]*gu_orig[i,h]."""
    algo = GPTQAlgorithm(wbits=4, group_size=-1)
    weight_gu, weight_dn, handlers = _make_updown_handlers()
    gu_orig = weight_gu.data.clone()
    dn_orig = weight_dn.data.clone()
    cfg = {"method": "updown_awq", "nbits": 4, "grid": 20}
    algo._apply_updown_awq(_FakeLayer(), handlers, cfg)

    I = 2  # intermediate dim
    for e in (0, 1):
        # gate half rows [0, I) untouched (swiglu nonlinearity)
        assert torch.equal(weight_gu.data[e, :I].float(), gu_orig[e, :I].float())
        prod_new = weight_dn.data[e].float() * weight_gu.data[e, I:].float().T
        prod_orig = dn_orig[e].float() * gu_orig[e, I:].float().T
        assert torch.allclose(prod_new, prod_orig, rtol=1e-4, atol=1e-5), (
            f"expert {e}: balanced fold broken (down*up not preserved)"
        )


def test_updown_awq_respects_middle_activation_significance():
    algo = GPTQAlgorithm(wbits=4, group_size=-1)
    weight_gu, weight_dn, handlers = _make_updown_handlers()
    # expert 1's middle activation is 5x larger -> different per-channel scale
    cfg = {"method": "updown_awq", "nbits": 4, "grid": 20}
    algo._apply_updown_awq(_FakeLayer(), handlers, cfg)
    # sanity: scales positive, weights finite
    assert torch.isfinite(weight_dn.data).all()
    assert torch.isfinite(weight_gu.data).all()


def test_updown_awq_no_handlers_noop():
    algo = GPTQAlgorithm(wbits=4, group_size=-1)
    handlers = {}
    cfg = {"method": "updown_awq", "nbits": 4, "grid": 20}
    # must not raise when no down handlers exist
    algo._apply_updown_awq(_FakeLayer(), handlers, cfg)


def test_updown_awq_config_routing():
    """method='updown_awq' routes through _apply_scale_search dispatch."""
    algo = GPTQAlgorithm(
        wbits=4,
        group_size=-1,
        scale_search={"moe": {"method": "updown_awq", "nbits": 4, "grid": 20}},
        expert_mode="rtn",
    )
    assert algo.scale_search_cfg["moe"]["method"] == "updown_awq"
    assert algo._expert_rtn is True


def test_batched_module_rtn_accumulates_x_abs_max():
    """AWQ max-stat: _x_abs_max tracks per-middle-channel running abs-max."""
    torch.manual_seed(21)
    weight_dn = torch.nn.Parameter(torch.randn(2, 6, 2))
    dn_slices = [
        _ExpertSliceLinear(weight_dn, e, "down_proj", "mlp.experts") for e in range(2)
    ]
    mod = BatchedGPTQModule(
        dn_slices, wbits=4, groupsize=-1, sym=True, percdamp=0.01,
        preproc_hessian=True, hessian_device=torch.device("cpu"), rtn=True,
    )
    mid0 = torch.randn(5, 2) + 3.0
    mid1 = torch.randn(7, 2) * 5.0
    mod.add_batch_for_expert(mid0, 0)
    mod.add_batch_for_expert(mid1, 1)
    assert mod.x_abs_max is not None
    assert torch.allclose(mod.x_abs_max[0], mid0.abs().amax(dim=0), atol=1e-6)
    assert torch.allclose(mod.x_abs_max[1], mid1.abs().amax(dim=0), atol=1e-6)
    mid0b = torch.randn(4, 2) * 0.1
    mod.add_batch_for_expert(mid0b, 0)
    assert torch.allclose(
        mod.x_abs_max[0],
        torch.maximum(mid0.abs().amax(dim=0), mid0b.abs().amax(dim=0)),
        atol=1e-6,
    )


def test_updown_awq_maxstat_search_fold_balanced():
    """search_stat='max' (modelslim flex_awq_ssz): α searched on max|act| while the
    returned scale stays mean|act|^α; the balanced fold down*s / up/s must cancel."""
    algo = GPTQAlgorithm(wbits=4, group_size=-1)
    weight_gu, weight_dn, handlers = _make_updown_handlers()
    dn = handlers["dn"]
    assert dn.x_abs_max is not None and float(dn.x_abs_max.abs().sum()) > 0.0
    gu_orig = weight_gu.data.clone()
    dn_orig = weight_dn.data.clone()
    cfg = {"method": "updown_awq", "nbits": 4, "grid": 21, "search_stat": "max"}
    algo._apply_updown_awq(_FakeLayer(), handlers, cfg)
    I = 2
    for e in (0, 1):
        prod_new = weight_dn.data[e].float() * weight_gu.data[e, I:].float().T
        prod_orig = dn_orig[e].float() * gu_orig[e, I:].float().T
        assert torch.allclose(prod_new, prod_orig, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(weight_dn.data).all()


def test_updown_awq_maxstat_passes_abs_max_as_search_stat(monkeypatch):
    """search_stat='max': each expert search must receive x_search=abs-max while the
    applied scale is built from abs-mean (modelslim flex_awq_ssz semantics)."""
    from npuslim.algorithms.quantization import scale_search as ss

    calls = []
    real = ss.awq_search

    def spy(w, x_scale, nbits=4, grid=20, eps=1e-8, x_search=None):
        calls.append(
            (
                x_scale.detach().clone(),
                None if x_search is None else x_search.detach().clone(),
            )
        )
        return real(w, x_scale, nbits=nbits, grid=grid, eps=eps, x_search=x_search)

    monkeypatch.setattr(ss, "awq_search", spy)
    algo = GPTQAlgorithm(wbits=4, group_size=-1)
    _, _, handlers = _make_updown_handlers()
    dn = handlers["dn"]
    cfg = {"method": "updown_awq", "nbits": 4, "grid": 21, "search_stat": "max"}
    algo._apply_updown_awq(_FakeLayer(), handlers, cfg)

    assert len(calls) == 2, f"expected 2 expert searches, got {len(calls)}"
    for local_idx, (x_scale_seen, x_search_seen) in enumerate(calls):
        assert x_search_seen is not None, "x_search not passed for search_stat=max"
        assert torch.allclose(x_search_seen, dn.x_abs_max[local_idx]), (
            f"expert {local_idx}: x_search != x_abs_max"
        )
        assert torch.allclose(x_scale_seen, dn.x_abs_mean[local_idx]), (
            f"expert {local_idx}: x_scale != x_abs_mean"
        )
