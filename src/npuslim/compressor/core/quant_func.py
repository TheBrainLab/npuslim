from typing import Tuple
import torch
from torch import nn


# ============================================================
# Legacy / non-GPTQ int8 weight quantization
# ============================================================


@torch.no_grad()
def quantize_weight_int(
    x: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # group-wise weight scale -> element-wise
    if scales.ndim == 2:
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)

    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)

    scales.div_(bnt)
    x.div_(scales).round_().clamp_(-bnt - 1, bnt)
    return x, scales


# ============================================================
# GPTQ-style weight quantizer
# ============================================================


class WeightQuantizer(nn.Module):
    """
    Unified weight quantizer for GPTQ-style algorithms.
    """

    def __init__(
        self,
        bits: int,
        method: str = "minmax",
        perchannel: bool = True,
        sym: bool = True,
    ):
        self.bits = bits
        self.method = method.lower()
        self.perchannel = perchannel
        self.sym = sym

        self.scale = None
        self.zero = None
        self.maxq = None

    # ------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------

    @torch.no_grad()
    def find_params(self, x: torch.Tensor):
        if self.method == "minmax":
            self._find_params_minmax(x)
        elif self.method == "rms":
            self.scale = None
            self.zero = None
            self.maxq = 2**self.bits - 1
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")

    @torch.no_grad()
    def _find_params_minmax(self, x: torch.Tensor):
        maxq = 2**self.bits - 1
        shape = x.shape
        device = x.device

        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=device)
        xmin = torch.minimum(x.min(dim=1)[0], tmp)
        xmax = torch.maximum(x.max(dim=1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(xmax, xmin.abs())
            xmin = -xmax

        zero_mask = (xmin == 0) & (xmax == 0)
        xmin[zero_mask] = -1.0
        xmax[zero_mask] = 1.0

        scale = (xmax - xmin) / maxq

        if self.sym:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        out_shape = [-1] + [1] * (len(shape) - 1)
        self.scale = scale.reshape(out_shape)
        self.zero = zero.reshape(out_shape)
        self.maxq = maxq

    # ------------------------------------------------------------
    # Float quantization (for analysis / forward)
    # ------------------------------------------------------------

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "minmax":
            assert self.ready()
            return self._quant_minmax(x)
        elif self.method == "rms":
            return self._quant_rms(x)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")

    @torch.no_grad()
    def _quant_minmax(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.round(x / self.scale) + self.zero
        q = torch.clamp(q, 0, self.maxq)
        return self.scale * (q - self.zero)

    @torch.no_grad()
    def _quant_rms(self, x: torch.Tensor) -> torch.Tensor:
        scale = 2.4 * x.square().mean().sqrt() + 1e-16
        maxq = 2**self.bits - 1

        q = (x / scale + 1.0) / 2.0
        q = torch.round(q * maxq).clamp(0, maxq)
        q = q / maxq * 2.0 - 1.0
        return q * scale

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def ready(self) -> bool:
        return self.method == "rms" or self.scale is not None

    def clear(self):
        self.scale = None
        self.zero = None
        self.maxq = None
