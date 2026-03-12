import torch
import torch.nn as nn
import transformers
from loguru import logger
from npuslim.compressor.core.base_hessian_module import BaseHessianModule
from npuslim.compressor.core.quant_func import compute_scales_with_zero, quantize_with_scale_zero


class GPTQModule(BaseHessianModule):
    """GPTQ quantization module for individual layers."""

    def __init__(self, *args, **kwargs):
        super(GPTQModule, self).__init__(*args, **kwargs)

        # Quantization parameters
        self.w_bits = getattr(self.config, "w_bits", 4)
        self.groupsize = getattr(self.config, "group_size", 128)
        self.blocksize = getattr(self.config, "blocksize", 128)
        self.actorder = getattr(self.config, "actorder", True)
        self.static_groups = getattr(self.config, "static_groups", True)
        self.sym = getattr(self.config, "sym", True)

        # Pre-computed scales and zeros (replaces WeightQuantizer state)
        self.scales = []
        self.zeros = []
        self.scale = None
        self.zero = None

    def fasterquant(self, **kwargs):
        """Execute GPTQ quantization algorithm."""
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        if self.actorder and not self.static_groups and self.groupsize != -1:
            logger.warning(
                "ActOrder enabled but static_groups is False. Forcing static_groups=True."
            )
            self.static_groups = True

        # Pre-compute scales/zeros for static groups (replaces quantizer.find_params)
        if self.static_groups and self.groupsize != -1:
            for i in range(0, self.columns, self.groupsize):
                scale, zero = compute_scales_with_zero(
                    W[:, i : i + self.groupsize],
                    bits=self.w_bits,
                    sym=self.sym,
                )
                self.scales.append(scale)
                self.zeros.append(zero)

        # For non-grouped quantization, compute scale/zero once
        if self.groupsize == -1:
            self.scale, self.zero = compute_scales_with_zero(
                W,
                bits=self.w_bits,
                sym=self.sym,
            )

        H = self.H
        if self.actorder:
            from npuslim.utils.backend import bh
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            # NOTE: Ascend AiCore does not support argsort with int32/int64 dtypes.
            # Cast to float32 to ensure high-performance execution on AiCore.
            invperm = torch.argsort(perm.float()) if bh.name == "npu" else torch.argsort(perm)
        else:
            perm = torch.arange(self.columns, device=W.device)
            invperm = perm

        Hinv = self.compute_hinv(H)
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                current_col = i1 + i

                # Get scale/zero for current position (replaces quantizer state access)
                if self.groupsize != -1:
                    if self.static_groups:
                        original_idx = perm[current_col]
                        group_idx = original_idx // self.groupsize
                        scale = self.scales[group_idx]
                        zero = self.zeros[group_idx]
                    else:
                        if current_col % self.groupsize == 0:
                            scale, zero = compute_scales_with_zero(
                                W[:, current_col : current_col + self.groupsize],
                                bits=self.w_bits,
                                sym=self.sym,
                            )
                            self.scales.append(scale)
                            self.zeros.append(zero)
                        else:
                            scale = self.scales[current_col // self.groupsize]
                            zero = self.zeros[current_col // self.groupsize]
                else:
                    scale = self.scale
                    zero = self.zero

                # Quantize (replaces quantizer.quantize)
                q = quantize_with_scale_zero(
                    w.unsqueeze(1),
                    scale,
                    zero,
                    bits=self.w_bits,
                ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # Build group index
        if self.groupsize == -1:
            g_idx = torch.zeros(self.columns, dtype=torch.int32, device="cpu")
        else:
            if self.static_groups:
                g_idx = (perm // self.groupsize).to(torch.int32).cpu()
            else:
                g_idx = (torch.arange(self.columns, device="cpu") // self.groupsize).to(
                    torch.int32
                )

        g_idx = g_idx.to(self.layer.weight.device)
        if self.actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        avg_loss = torch.sum(Losses).item() / self.nsamples
        W_orig = self.layer.weight.data.float()
        norm_loss = torch.norm(Q - W_orig).item()
        self.print_log(avg_loss=avg_loss, norm_loss=norm_loss)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Prepare final scales and zeros
        if self.groupsize != -1:
            final_scale = torch.cat(self.scales, dim=1)
            final_zero = torch.cat(self.zeros, dim=1)
        else:
            final_scale = self.scale
            final_zero = self.zero

        self.postproc()
        self.free()

        return final_scale.cpu(), final_zero.cpu(), g_idx
