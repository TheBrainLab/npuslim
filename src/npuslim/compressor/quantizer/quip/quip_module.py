"""
QuIP (Quantization with Incoherence Processing) Module.

This implementation follows the original QuIP paper using:
1. Balance/LDLQ algorithm for quantization
2. Incoherence preprocessing (rescale + projection)
3. Fake quantization (float16 output, no packing)

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
    - Outputs float16 weights (fake quantization)
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

        # Scale/zero for minmax mode (computed in fasterquant)
        self.scale = None
        self.zero = None
        self.maxq = 2**self.w_bits - 1

    def fasterquant(self, **kwargs):
        """
        Execute QuIP quantization using LDLQ/Vector Balance algorithm.

        Following original QuIP's Balance.fasterquant():
        1. Get weight and Hessian
        2. Apply quantize_weight_ldlq
        3. Update layer weight
        4. Apply postproc (reverse preprocessing)

        Returns dummy scale/zero/g_idx for framework compatibility.
        """
        # Get weight
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()

        full_w = w.clone()
        tick = time.time()

        # Only compute scale/zero for minmax mode
        if self.quant_func == "minmax":
            if self.scale is None:
                self.scale, self.zero = compute_scales_with_zero(
                    w, bits=self.w_bits, sym=False
                )

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

        # Apply LDLQ quantization
        quant_w = quantize_weight_ldlq(w=w, H=H, config=ldlq_config)

        # Update layer weight
        if isinstance(self.layer, transformers.Conv1D):
            quant_w = quant_w.t()

        self.layer.weight.data = quant_w.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Compute error metrics BEFORE postproc (in preprocessed space)
        # This matches the original QuIP behavior
        quant_w_preproc = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            quant_w_preproc = quant_w_preproc.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            quant_w_preproc = quant_w_preproc.t()

        norm_loss = torch.norm(quant_w_preproc - full_w).item()
        self.error = ((full_w - quant_w_preproc) @ H.type(torch.float) @ (full_w - quant_w_preproc).T).trace().item()
        self.Hmag = self.H.max().item()

        # Post-processing: reverse incoherence processing
        self.postproc()

        # Timing
        self.time = time.time() - tick

        self.print_log(avg_loss=self.error / self.nsamples, norm_loss=norm_loss)

        self.free()

        # Return dummy values for framework compatibility
        # (fake quantization doesn't need real scale/zero)
        dummy_scale = torch.ones(self.rows, 1)
        dummy_zero = torch.zeros(self.rows, 1)
        dummy_g_idx = torch.zeros(self.columns, dtype=torch.int32)

        return dummy_scale.cpu(), dummy_zero.cpu(), dummy_g_idx

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
