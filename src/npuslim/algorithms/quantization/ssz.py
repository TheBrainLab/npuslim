"""SSZ (Scan-Scale-Zero) per-channel scale refinement for weight quantization.

Ported from msmodelslim `core/quantizer/impl/ssz.py` (Mulan PSL v2, MindStudio
official reference). The algorithm iteratively refines the per-output-channel
scale to minimize the weight quantization MSE:

  1. start from an initial scale (e.g. minmax);
  2. quantize, then compute the least-squares optimal scale for the current
     quantized values: s = sum(w*q) / sum(q^2);
  3. re-quantize with the new scale, compare per-channel MSE, keep the best
     scale greedily;
  4. repeat until convergence or `step` iterations.

This module implements the **symmetric** branch only (offset == 0 in the signed
quant grid), which matches the npuslim W4A8/W8A8 per-channel symmetric pack
protocol (q = clamp(round(w/s) + zero, 0, maxq), zero = (maxq+1)/2, dequant =
s*(q - zero)). It is a pure-torch, vectorized variant that operates on an
arbitrary number of rows ([N, in]) at once, so it works both per-expert and
across a batched expert chunk.

Reference file:
    /data/yult/llm_inference/msmodelslim/msmodelslim/core/quantizer/impl/ssz.py
"""

from __future__ import annotations

import torch

SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10
SCALE_SEARCH_MIN_SCALE = 1e-30
SCALE_SEARCH_DEFAULT_STEP = 10


@torch.no_grad()
def ssz_calculate_scale(
    w: torch.Tensor,
    initial_scale: torch.Tensor,
    bits: int = 4,
    step: int = SCALE_SEARCH_DEFAULT_STEP,
    converge_threshold: float = SCALE_SEARCH_CONVERGE_THRESHOLD,
    min_scale: float = SCALE_SEARCH_MIN_SCALE,
) -> torch.Tensor:
    """Iteratively refine per-row scale via least-squares (symmetric SSZ).

    Args:
        w: [N, in] float weight matrix (N rows = output channels to refine)
        initial_scale: [N, 1] initial per-row scale (e.g. minmax)
        bits: quantization bit width (4 or 8)
        step: max SSZ iterations (modelslim production yaml uses 10)
        converge_threshold: relative/absolute MSE improvement threshold
        min_scale: clamp floor to avoid division by zero

    Returns:
        [N, 1] refined per-row scale
    """
    w = w.float()
    maxq = 2 ** int(bits) - 1
    zero = (maxq + 1) / 2  # symmetric mid offset (matches compute_scales_with_zero)

    s = initial_scale.float().clamp_min(min_scale)
    q = torch.clamp(torch.round(w / s) + zero, 0, maxq)
    best_scale = s.clone()
    best_mse = torch.mean(
        torch.pow(torch.abs(w - s * (q - zero)), 2), dim=1, keepdim=True
    )

    for _ in range(int(step)):
        v = q - zero  # signed quantized values [N, in]
        # least-squares optimal scale for the current quantized values
        current_scale = (torch.sum(w * v, dim=1, keepdim=True) /
                         torch.sum(v * v, dim=1, keepdim=True).clamp(min=min_scale))
        current_scale = current_scale.clamp_min(min_scale)
        s = current_scale
        q = torch.clamp(torch.round(w / s) + zero, 0, maxq)
        current_mse = torch.mean(
            torch.pow(torch.abs(w - s * (q - zero)), 2), dim=1, keepdim=True
        )

        # convergence: every channel satisfies a relative or absolute threshold
        mask1 = (best_mse - current_mse) / best_mse.clamp(min=1e-4) < converge_threshold
        mask2 = torch.abs(best_mse - current_mse) < converge_threshold
        if (torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))).sum() == 0:
            break

        # greedy per-channel keep-best
        mask = (current_mse < best_mse).to(torch.int32)
        best_mse = best_mse * (1 - mask) + current_mse * mask
        best_scale = best_scale * (1 - mask) + current_scale * mask

    return best_scale
