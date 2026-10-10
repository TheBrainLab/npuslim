"""Unit tests for OmniQuant-style activation-aware scale refinement (ssz.omni_refine_scale).

Covers §6.9.2#3 (D0 probe): gradient-refined per-row scale with a true-quantizer STE
must reduce the activation-weighted forward MSE E||ΔW x||² relative to the weight-only
clip+SSZ warm start on synthetic outlier-heavy data, and must degrade safely
(steps=0 / no activations / isotropic activations).
"""

import torch

from npuslim.algorithms.quantization.ssz import clip_search_scale, omni_refine_scale


def _make_outlier_weight(rows: int, cols: int, outlier_cols, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g)
    w[:, outlier_cols] *= 12.0
    return w


def _make_correlated_activations(n: int, cols: int, outlier_cols, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(n, cols, generator=g)
    x[:, outlier_cols] *= 8.0  # activation outliers on the same channels
    return x


def _act_mse(w: torch.Tensor, s: torch.Tensor, x_cm: torch.Tensor) -> torch.Tensor:
    """E||ΔW x||² via the low-rank form ‖(W − Q(s)) Xᵀ‖²_F / n (true quantizer)."""
    q = torch.clamp(torch.round(w / s), -8, 7) * s
    y = (w - q) @ x_cm
    return (y * y).mean()


class TestOmniRefineScale:
    def test_reduces_activation_mse_when_outliers_align(self):
        """Given: weight and activation outliers on the same channels.
        When: omni_refine_scale refines the clip+SSZ warm start.
        Then: activation-weighted MSE strictly decreases with finite positive scale.
        """
        rows, cols, n = 64, 48, 256
        outlier_cols = torch.tensor([3, 11, 29, 40])
        w = _make_outlier_weight(rows, cols, outlier_cols, 7)
        x_cm = _make_correlated_activations(n, cols, outlier_cols, 7).t().contiguous()
        s0 = clip_search_scale(
            w, w.abs().amax(dim=1, keepdim=True) / 7.0, bits=4, lo=0.5, hi=1.0, steps=20
        )
        before = _act_mse(w, s0, x_cm)

        s1 = omni_refine_scale(w, s0, x_cm, bits=4, steps=80, lr=0.03)

        assert torch.isfinite(s1).all()
        assert (s1 > 0).all()
        after = _act_mse(w, s1, x_cm)
        assert after < before

    def test_respects_log_scale_range(self):
        """Given: u_range=1.0 (±e-fold around the warm start).
        When: refinement runs with aggressive lr.
        Then: every refined row stays within the allowed log-scale band.
        """
        rows, cols = 32, 24
        outlier_cols = torch.tensor([2, 9])
        w = _make_outlier_weight(rows, cols, outlier_cols, 13)
        x_cm = _make_correlated_activations(128, cols, outlier_cols, 13).t().contiguous()
        s0 = clip_search_scale(
            w, w.abs().amax(dim=1, keepdim=True) / 7.0, bits=4, lo=0.5, hi=1.0, steps=20
        )

        s1 = omni_refine_scale(w, s0, x_cm, bits=4, steps=50, lr=0.05, u_range=1.0)

        ratio = (s1 / s0).log()
        assert ratio.abs().max() <= 1.0 + 1e-4

    def test_zero_steps_returns_initial(self):
        """Given: steps=0.
        When: omni_refine_scale is called.
        Then: the initial scale is returned unchanged (opt-out path).
        """
        w = torch.randn(8, 16)
        s0 = torch.rand(8, 1) + 0.5
        x_cm = torch.randn(16, 32)

        s1 = omni_refine_scale(w, s0, x_cm, bits=4, steps=0)

        assert torch.equal(s1, s0)

    def test_empty_activations_returns_initial(self):
        """Given: an expert with no routed tokens (empty activation capture).
        When: omni_refine_scale is called.
        Then: the initial scale is returned unchanged and no error is raised.
        """
        w = torch.randn(8, 16)
        s0 = torch.rand(8, 1) + 0.5
        x_cm = torch.empty(16, 0)

        s1 = omni_refine_scale(w, s0, x_cm, bits=4, steps=40)

        assert torch.equal(s1, s0)

    def test_isotropic_activations_do_not_increase_mse(self):
        """Given: isotropic activations (H ∝ I, weight-only objective already near-optimal).
        When: refinement runs.
        Then: activation-weighted MSE does not increase (safe warm-start property).
        """
        rows, cols, n = 32, 24, 256
        g = torch.Generator().manual_seed(17)
        w = torch.randn(rows, cols, generator=g)
        x_cm = torch.randn(n, cols, generator=g).t().contiguous()
        s0 = clip_search_scale(
            w, w.abs().amax(dim=1, keepdim=True) / 7.0, bits=4, lo=0.5, hi=1.0, steps=20
        )
        before = _act_mse(w, s0, x_cm)

        s1 = omni_refine_scale(w, s0, x_cm, bits=4, steps=50, lr=0.02)

        after = _act_mse(w, s1, x_cm)
        assert after <= before * 1.001
