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
        elif self.method == "rms_symmetric":
            self._find_params_rms_symmetric(x)
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

    @torch.no_grad()
    def _find_params_rms_symmetric(self, x: torch.Tensor):
        """
        RMS symmetric quantization method compatible with GPTQ packing format.

        For symmetric quantization:
        - scale = 2.4 * sqrt(mean(x^2)) per output channel
        - zero = maxq / 2 (symmetric zero point)

        This produces scale/zero compatible with GPTQQuantLinear.pack().
        """
        maxq = 2**self.bits - 1
        shape = x.shape

        if self.perchannel:
            x_flat = x.flatten(1)  # [out_features, in_features]
        else:
            x_flat = x.flatten().unsqueeze(0)

        # RMS scale: 2.4 * sqrt(mean(x^2)) per output channel (row-wise)
        scale = 2.4 * x_flat.square().mean(dim=1).sqrt() + 1e-16

        # Symmetric zero point: maxq / 2
        zero = torch.full_like(scale, maxq / 2)

        out_shape = [-1] + [1] * (len(shape) - 1)
        self.scale = scale.reshape(out_shape)
        self.zero = zero.reshape(out_shape)
        self.maxq = maxq

    # ------------------------------------------------------------
    # Float quantization (for analysis / forward)
    # ------------------------------------------------------------

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
        """
        Quantize input tensor.

        Args:
            x: Input tensor to quantize
            scale: Pre-computed scale for RMS method. If None, will compute internally.
        """
        if self.method == "minmax":
            assert self.ready()
            return self._quant_minmax(x)
        elif self.method == "rms":
            return self._quant_rms(x, scale)
        elif self.method == "rms_symmetric":
            assert self.ready()
            return self._quant_rms_symmetric(x)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")

    @torch.no_grad()
    def _quant_minmax(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.round(x / self.scale) + self.zero
        q = torch.clamp(q, 0, self.maxq)
        return self.scale * (q - self.zero)

    @torch.no_grad()
    def _quant_rms(self, x: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
        """
        RMS quantization method.

        Args:
            x: Input tensor to quantize
            scale: Pre-computed RMS scale. If None, will compute internally.
        """
        maxq = self.maxq if self.maxq is not None else (2**self.bits - 1)

        # Use provided scale or compute internally
        if scale is not None:
            scale = scale.to(x.device)
        else:
            scale = self.get_rms_scale(x)

        q = (x / scale + 1.0) / 2.0
        q = torch.round(q * maxq).clamp(0, maxq)
        q = q / maxq * 2.0 - 1.0
        return q * scale

    @torch.no_grad()
    def _quant_rms_symmetric(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS symmetric quantization compatible with GPTQ packing format.

        Uses clamp-before-round (LDL equivalent) for better numerical stability.

        Quantization formula (symmetric):
            q = clamp(round(x / scale) + zero, 0, maxq)
            x_dequant = scale * (q - zero)
        """
        q = torch.round(x / self.scale) + self.zero
        q = torch.clamp(q, 0, self.maxq)
        return self.scale * (q - self.zero)

    @torch.no_grad()
    def get_rms_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get RMS scale for a given input tensor.
        
        RMS scale formula: scale = 2.4 * sqrt(mean(x^2))
        
        Args:
            x: Input tensor of shape [..., features]
        
        Returns:
            Scale tensor. If perchannel=True, returns scale per output channel.
        """
        if self.perchannel:
            # Per-channel scale: compute mean over all dimensions except the last (feature dim)
            # For weight tensor of shape [out_features, in_features]
            # We want scale per output channel (row)
            return 2.4 * x.square().mean(dim=list(range(1, x.ndim))).sqrt() + 1e-16
        else:
            # Single scale for the entire tensor
            return 2.4 * x.square().mean().sqrt() + 1e-16

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def ready(self) -> bool:
        # rms method computes scale dynamically, doesn't need pre-computed scale
        # rms_symmetric and minmax require pre-computed scale/zero
        return self.method == "rms" or self.scale is not None

    def clear(self):
        self.scale = None
        self.zero = None
        self.maxq = None
