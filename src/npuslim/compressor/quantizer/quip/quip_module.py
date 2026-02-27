"""
QuIP (Quantization with Incoherence Processing) Module.

This implementation follows the original QuIP paper using:
1. Balance/LDLQ algorithm for quantization
2. Incoherence preprocessing (rescale + projection)
3. Supports both fake quantization and real quantization

Reference: https://github.com/Cornell-RelaxML/QuIP
"""

import time
import torch
import torch.nn as nn
import transformers

from npuslim.compressor.core.base_hessian_module import BaseHessianModule
from npuslim.compressor.core.quant_func import compute_scales_with_zero
from .vector_balance import quantize_weight_ldlq, LDLQConfig


class QuIPModule(BaseHessianModule):
    """
    QuIP Module using LDLQ/Vector Balance algorithm.

    Following original QuIP's Balance class (bal.py):
    - Uses quantize_weight_ldlq for quantization
    - Supports quant_func="minmax" or "rms"
    - Returns quantization parameters for real quantization
    """

    def __init__(self, *args, **kwargs):
        super(QuIPModule, self).__init__(*args, **kwargs)

        # QuIP parameters
        self.w_bits = getattr(self.config, "w_bits", 4)
        self.npasses = getattr(self.config, "npasses", 0)  # greedy passes
        self.unbiased = getattr(self.config, "unbiased", False)
        self.quant_func = getattr(self.config, "quant_func", "rms")  # "minmax" or "rms"
        self.ldlq_method = getattr(self.config, "ldlq_method", "ldlq")
        self.blocksize = getattr(self.config, "blocksize", 128)
        self.fake_quant = getattr(self.config, "fake_quant", True)

        # Scale/zero for minmax mode (computed in fasterquant)
        self.scale = None
        self.zero = None
        self.maxq = 2**self.w_bits - 1

        # RMS scale (computed in fasterquant for rms mode)
        self.scale_rms = None

    def fasterquant(self, **kwargs):
        """
        Execute QuIP quantization using LDLQ/Vector Balance algorithm.

        Following original QuIP's Balance.fasterquant():
        1. Get weight and Hessian
        2. Apply quantize_weight_ldlq
        3. Update layer weight
        4. Apply postproc (reverse preprocessing) - only for fake_quant

        Returns:
            For fake_quant: dummy scale/zero/g_idx
            For real_quant: dict with w_int, scales, zeros, scaleWH, proj_seeds
        """
        # Get weight
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()

        full_w = w.clone()
        tick = time.time()

        # Compute scale/zero for minmax mode, scale_rms for rms mode
        if self.quant_func == "minmax":
            if self.scale is None:
                self.scale, self.zero = compute_scales_with_zero(
                    w, bits=self.w_bits, sym=False
                )
        else:  # rms
            # Compute RMS scale (same as in quantize_weight_ldlq)
            self.scale_rms = 2.4 * w.square().mean().sqrt() + 1e-16

        # Get Hessian
        H = self.H.data.clone()

        # Build LDLQConfig
        ldlq_config = LDLQConfig(
            nbits=self.w_bits,
            quant_func=self.quant_func,
            ldlq_method=self.ldlq_method,
            npasses=self.npasses,
            unbiased=self.unbiased,
            blocksize=self.blocksize,
            scale=self.scale,
            zero=self.zero,
        )

        # Apply LDLQ quantization (returns dequantized float weights and rounded integers)
        quant_w, w_int = quantize_weight_ldlq(w=w, H=H, config=ldlq_config, return_int=True)

        # Update layer weight
        if isinstance(self.layer, transformers.Conv1D):
            quant_w = quant_w.t()

        self.layer.weight.data = quant_w.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Compute error metrics BEFORE postproc (in preprocessed space)
        quant_w_preproc = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            quant_w_preproc = quant_w_preproc.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            quant_w_preproc = quant_w_preproc.t()

        norm_loss = torch.norm(quant_w_preproc - full_w).item()
        self.error = (
            (
                (full_w - quant_w_preproc)
                @ H.type(torch.float)
                @ (full_w - quant_w_preproc).T
            )
            .trace()
            .item()
        )
        self.Hmag = self.H.max().item()

        # Post-processing: reverse incoherence processing
        # For fake_quant: weights stay as float16 in the layer
        # For real_quant: we still need to apply postproc so subsequent layers get correct activations
        # (The real quant params w_int are already saved before postproc)
        self.postproc()

        # Timing
        self.time = time.time() - tick

        self.print_log(avg_loss=self.error / self.nsamples, norm_loss=norm_loss)

        # Collect preprocessing parameters BEFORE free()
        result = self._collect_quant_params(w_int)

        self.free()

        return result

    def _extract_int_weights(self, w: torch.Tensor, quant_w: torch.Tensor) -> torch.Tensor:
        """
        Extract integer weights from the quantization result.

        Args:
            w: Original weights [out, in] (in preprocessed space)
            quant_w: Dequantized weights [out, in]

        Returns:
            Integer weights [out, in] in range [0, maxq]
        """
        if self.quant_func == "minmax":
            # minmax: w_int = round(w / scale) + zero
            if self.scale is None or self.zero is None:
                raise ValueError("scale and zero must be computed for minmax mode")
            w_int = torch.clamp(
                torch.round(w / self.scale) + self.zero,
                0, self.maxq
            )
        else:  # rms
            # rms: w_normalized = w / scale_rms
            #      w_int = round((w_normalized + 1) / 2 * maxq)
            if self.scale_rms is None:
                raise ValueError("scale_rms must be computed for rms mode")
            w_normalized = w / self.scale_rms
            w_int = torch.clamp(
                torch.round(((w_normalized + 1) / 2) * self.maxq),
                0, self.maxq
            )

        return w_int.to(torch.int32)

    def _collect_quant_params(self, w_int: torch.Tensor) -> dict:
        """
        Collect all parameters needed for real quantization.

        Args:
            w_int: Integer weights [out, in]

        Returns:
            dict with all parameters for QuIPLinear.pack()
        """
        # Get preprocessing parameters from BaseHessianModule
        scaleWH = getattr(self, "scaleWH", None)
        proj_seed_u = getattr(self, "proj_seed_u", 0)
        proj_seed_v = getattr(self, "proj_seed_v", 0)
        proj_mode = getattr(self.config, "preproc_proj_mode", 2)

        # Prepare scales and zeros based on quant_func
        if self.quant_func == "minmax":
            if self.scale is None or self.zero is None:
                raise ValueError("scale and zero must be computed for minmax mode")
            scales = self.scale.half()  # [out, 1]
            zeros = self.zero.half()    # [out, 1]
        else:  # rms
            if self.scale_rms is None:
                raise ValueError("scale_rms must be computed for rms mode")
            scales = self.scale_rms.half().view(1)  # scalar
            zeros = None

        # Handle scaleWH
        if scaleWH is None:
            scaleWH = torch.ones(self.columns, dtype=torch.float32)
        else:
            scaleWH = scaleWH.float()

        return {
            "w_int": w_int.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu() if zeros is not None else None,
            "scaleWH": scaleWH.cpu(),
            "proj_seed_u": proj_seed_u,
            "proj_seed_v": proj_seed_v,
            "proj_mode": proj_mode,
            "bias": self.layer.bias.data.cpu() if self.layer.bias is not None else None,
        }

    def print_log(self, avg_loss, norm_loss):
        """Print quantization statistics."""
        w_shape = self.layer.weight.shape
        w_out, w_in = w_shape[0], w_shape[1]
        label_width = 25
        logger_info = [
            f"{'Layer Shape:':<{label_width}} [Out={w_out}, In={w_in}]",
            f"{'Hessian Error:':<{label_width}} {avg_loss:.6f}",
            f"{'Norm Loss (L2):':<{label_width}} {norm_loss:.6f}",
            f"{'Hessian Max:':<{label_width}} {self.Hmag:.6f}",
            f"{'Time:':<{label_width}} {self.time:.2f}s",
        ]
        from loguru import logger

        for line in logger_info:
            logger.info(line)
