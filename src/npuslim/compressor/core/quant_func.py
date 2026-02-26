# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantization functions for weight compression.

This module provides functional interfaces for weight quantization,
replacing the previous WeightQuantizer class with simpler, more explicit functions.
"""

from typing import Tuple
import torch


# ============================================================
# Legacy / non-GPTQ int8 weight quantization
# ============================================================


@torch.no_grad()
def quantize_weight_int(
    x: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-wise int8 weight quantization."""
    if scales.ndim == 2:
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)

    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)

    scales.div_(bnt)
    x.div_(scales).round_().clamp_(-bnt - 1, bnt)
    return x, scales


# ============================================================
# MinMax quantization functions (for GPTQ)
# ============================================================


@torch.no_grad()
def compute_scales_with_zero(
    x: torch.Tensor,
    bits: int = 4,
    sym: bool = True,
    perchannel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scale and zero for minmax quantization.

    Reference: AngelSlim implementation

    Args:
        x: Weight tensor [out_features, in_features]
        bits: Quantization bitwidth
        sym: Whether to use symmetric quantization
        perchannel: Whether to compute per-channel scales

    Returns:
        scale: Quantization scale [out_features, 1]
        zero: Zero point [out_features, 1]
    """
    maxq = 2**bits - 1
    shape = x.shape

    if perchannel:
        x = x.flatten(1)
    else:
        x = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]

    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)

    out_shape = [-1] + [1] * (len(shape) - 1)
    return scale.reshape(out_shape), zero.reshape(out_shape)


@torch.no_grad()
def quantize_with_scale_zero(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """
    Quantize weights using pre-computed scale and zero.

    Args:
        w: Weight tensor to quantize
        scale: Pre-computed scale
        zero: Pre-computed zero point
        bits: Quantization bitwidth

    Returns:
        Quantized weight tensor (fake quantization, float)
    """
    maxq = 2**bits - 1
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


# ============================================================
# RMS quantization function (for QuIP)
# ============================================================


@torch.no_grad()
def quantize_rms(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    RMS quantization with dynamic scale computation.

    This is used by QuIP algorithm. Scale is computed dynamically
    as 2.4 * sqrt(mean(w^2)).

    Args:
        w: Weight tensor to quantize
        bits: Quantization bitwidth

    Returns:
        Quantized weight tensor (float16)
    """
    maxq = 2**bits - 1
    scale = 2.4 * w.square().mean().sqrt() + 1e-16

    q = (w / scale + 1.0) / 2.0
    q = torch.round(q * maxq).clamp(0, maxq)
    q = q / maxq * 2.0 - 1.0

    return (q * scale).half()
