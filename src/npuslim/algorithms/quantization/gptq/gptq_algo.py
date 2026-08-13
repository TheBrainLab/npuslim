"""Chunk-wise GPTQ algorithm for compressor task.

Design goal:
- Strict chunk lifecycle: calibrate -> quantize -> pack inside each process_chunk call.
- No full model load requirement.
- Keep GPTQ core math/packing behavior aligned with v1 implementation.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from npuslim.algorithms.quantization.hessian import (
    BaseHessianAlgorithm,
    BaseHessianModule,
    _get_child_module,
    _is_transformers_conv1d,
    compute_scales_with_zero,
    quantize_with_scale_zero,
)
from npuslim.core import AlgorithmRegistry
from npuslim.core.backend import bh


class _ExpertSliceLinear(nn.Linear):
    """nn.Linear wrapping a 2D view of a 3D Parameter for per-expert GPTQ.

    Inherits nn.Linear so BaseHessianModule.add_batch's isinstance(self.layer, nn.Linear)
    check works correctly. The weight is a view into the 3D Parameter (no clone),
    so no extra GPU memory is consumed.

    After fasterquant (which replaces .weight.data), call write_back() to copy
    the result back to the 3D Parameter.

    Not registered in the model's module tree - used only as a weight container
    for GPTQModule. Hessian collection is handled by the expert collection hook
    in BaseHessianAlgorithm._collect_statistics.
    """

    def __init__(
        self,
        weight_3d: nn.Parameter,
        expert_idx: int,
        proj_type: str,
        experts_module_path: str,
    ):
        # Skip nn.Linear.__init__ to avoid creating a new weight tensor
        nn.Module.__init__(self)
        # View into the 3D Parameter (no clone, no extra memory)
        self.weight = nn.Parameter(weight_3d.data[expert_idx], requires_grad=False)
        self.bias = None
        self._weight_3d = weight_3d
        self._expert_idx = expert_idx
        self._proj_type = proj_type
        self._experts_module_path = experts_module_path
        self._is_expert_slice = True

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def write_back(self) -> None:
        """Copy the (possibly modified) weight back to the 3D Parameter.

        The 3D Parameter may be on CPU (offloaded to save GPU memory),
        while the weight may be on GPU after fasterquant.
        """
        self._weight_3d.data[self._expert_idx] = self.weight.data.to(
            self._weight_3d.device
        )


class BatchedGPTQModule:
    """Batched GPTQ for CH experts at once, using torch.bmm for column updates.

    Reduces Python fasterquant loop from E to E/CH (e.g. 256 -> 32 for CH=8).
    Each batched fasterquant call processes CH experts simultaneously:
    - Weight:  [CH, out, in]
    - Hessian: [CH, in, in] (on CPU, moved to GPU for fasterquant)
    - hinv:    [CH, in, in]
    - Column update: torch.bmm([CH,out,1], [CH,1,count-i]) -> [CH,out,count-i]

    Memory: CH * (weight_fp32 + hessian + hinv) ~ CH * 350MB for GLM-5.
    For CH=8: ~2.8GB, well within the 32GB - 20GB weights = 12GB budget.

    Note: actorder is disabled in batched mode (per-expert permutation makes
    group lookup impractical). preproc_hessian=True still handles dead columns.
    """

    def __init__(
        self,
        slice_linears: List[_ExpertSliceLinear],
        *,
        wbits: int = 4,
        groupsize: int = 128,
        blocksize: int = 128,
        sym: bool = True,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        hessian_device=None,
    ):
        self._slice_linears = slice_linears
        self._CH = len(slice_linears)
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.blocksize = int(blocksize)
        self.sym = bool(sym)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self._hdev = hessian_device if hessian_device is not None else slice_linears[0].weight.device

        w0 = slice_linears[0].weight.data
        self.dev = w0.device
        self.outfeatures = w0.shape[0]
        self.infeatures = w0.shape[1]

        # Hessian: accumulated on CPU to avoid GPU OOM with 256 experts.
        # Each batch's contrib (X@X.T) is computed on GPU and immediately
        # transferred to CPU via .to(self._hdev). This is the same pattern
        # as BaseHessianModule.add_batch.
        self.H = torch.zeros(self._CH, self.infeatures, self.infeatures, device=self._hdev)
        self.nsamples = torch.zeros(self._CH, device=self._hdev)
        self.preproc_done = False
        self.last_metrics: Dict[str, float] = {}
        self._hinv_fallback: Optional[str] = None
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.scales: List[torch.Tensor] = []
        self.zeros: List[torch.Tensor] = []

        # Metadata for expert collection hook
        self._is_batched_expert = True
        self._expert_start_idx = slice_linears[0]._expert_idx
        self._proj_type = slice_linears[0]._proj_type
        self._experts_module_path = slice_linears[0]._experts_module_path

    @property
    def layer(self):
        """Compatibility: return first slice for pack access."""
        return self._slice_linears[0]

    def add_batch_for_expert(self, inp: torch.Tensor, local_idx: int) -> None:
        """Accumulate Hessian contribution on CPU.

        Computes X@X.T on GPU, then immediately transfers to CPU to avoid
        the 38 GB GPU memory that 32×BatchedGPTQModule GPU Hessian accumulators would require.
        PCIe transfer overhead is acceptable for 256-expert MoE layers.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().to(self.dev, dtype=torch.float32)  # [in, batch]
        batch = inp.shape[1]
        contrib = inp.matmul(inp.t())  # [in, in] on GPU
        self.H[local_idx] += contrib.to(self._hdev)  # GPU -> CPU transfer
        self.nsamples[local_idx] += batch

    def preproc(self) -> None:
        # Hessian scaling: H = (2 / nsamples) * Σ(Xi @ Xi.T)
        ns = self.nsamples.clamp(min=1).to(self._hdev).view(-1, 1, 1)  # [CH, 1, 1]
        self.H = (2.0 / ns * self.H).to(self._hdev)  # [CH, in, in] on CPU

        if self.preproc_hessian:
            h = self.H.clone()
            dead = torch.diagonal(h, dim1=-2, dim2=-1) == 0  # [CH, in]
            self._dead_mask = dead  # [CH, in] — store for later use in fasterquant
            dead_mask_3d = dead.diag_embed().float()
            h = h + dead_mask_3d
            damp = self.percdamp * torch.mean(torch.diagonal(h, dim1=-2, dim2=-1))
            eye = torch.eye(self.infeatures, device=h.device).unsqueeze(0)
            h = h + damp * eye
            self.H = h.to(self.H.dtype)
        self.preproc_done = True

    def _build_weight_batch(self) -> torch.Tensor:
        """Build batched weight tensor [CH, out, in] on demand.

        Reads directly from the 3D Parameter (not the stale slice view)
        so that if the 3D Parameter was moved to CPU, only the current
        CH slice is transferred to GPU. This saves ~19 GB of peak VRAM.
        """
        if not hasattr(self, '_w_batch') or self._w_batch is None:
            self._w_batch = torch.stack(
                [sl._weight_3d.data[sl._expert_idx] for sl in self._slice_linears], dim=0
            ).to(self.dev)
        return self._w_batch

    def _compute_hinv_batched(self, hessian: torch.Tensor) -> torch.Tensor:
        # Compute Cholesky on GPU when available for 10-50x speedup.
        # At this point 3D MoE parameters have been moved to CPU, so GPU has
        # ample free memory. The Hessian [CH, in, in] is ~1.15 GB for CH=8,
        # well within budget. hinv is kept on CPU afterwards so fasterquant
        # only moves block-sized slices [CH, blocksize, in] to GPU on demand.
        use_gpu = self.dev.type == "cuda" and torch.cuda.is_available()
        h = hessian.to(self.dev) if use_gpu else (hessian.to("cpu") if hessian.device.type != "cpu" else hessian)
        step = 0.1
        current_percdamp = 0.0 if self.preproc_hessian else float(self.percdamp)
        hinv = None
        while current_percdamp < 10.0:
            try:
                h_try = h.clone()
                if current_percdamp > 0:
                    damp = current_percdamp * torch.mean(
                        torch.diagonal(h_try, dim1=-2, dim2=-1)
                    )
                    if damp.item() == 0:
                        damp = 1e-5
                    eye = torch.eye(self.infeatures, device=h_try.device).unsqueeze(0)
                    h_try = h_try + damp * eye
                chol = torch.linalg.cholesky(h_try)
                inv_chol = torch.cholesky_inverse(chol)
                hinv = torch.linalg.cholesky(inv_chol, upper=True)
                break
            except (RuntimeError, torch._C._LinAlgError):
                if current_percdamp == 0:
                    current_percdamp = float(self.percdamp)
                else:
                    current_percdamp += step
        if hinv is None:
            # Fallback: use pseudo-inverse for degenerate Hessians
            logger.warning(f"[GPTQ] Batched Cholesky failed after max damping, using pseudo-inverse")
            self._hinv_fallback = "pinv"
            try:
                hinv = torch.linalg.pinv(h)
            except Exception:
                # Last resort: identity matrix (skip quantization for this batch)
                logger.warning(f"[GPTQ] Pseudo-inverse also failed, using identity (no quantization)")
                self._hinv_fallback = "identity"
                hinv = torch.eye(h.shape[-1], device=h.device).unsqueeze(0).expand_as(h)
        else:
            self._hinv_fallback = None
        # Move hinv to CPU: fasterquant moves only block-sized slices to GPU
        # on demand, saving 576 MB (CH=4) or 1.15 GB (CH=8) of peak GPU memory.
        if hinv.device.type != "cpu":
            hinv = hinv.to("cpu")
        return hinv

    def fasterquant(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        CH, out, in_ = self._CH, self.outfeatures, self.infeatures
        w = self._build_weight_batch().float().clone()  # [CH, out, in]

        # Apply dead column masking from preproc
        if self.preproc_hessian:
            dead = self._dead_mask.to(self.dev)  # [CH, in]
            w = w.masked_fill(dead.unsqueeze(1), 0)

        # Static scales/zeros (actorder disabled for batched mode)
        self.scales = []
        self.zeros = []
        if self.groupsize != -1:
            for i in range(0, in_, self.groupsize):
                chunk = w[:, :, i : i + self.groupsize]
                s, z = compute_scales_with_zero(
                    chunk.reshape(CH * out, self.groupsize), self.wbits, self.sym
                )
                self.scales.append(s.reshape(CH, out, 1))
                self.zeros.append(z.reshape(CH, out, 1))

        hessian = self.H  # [CH, in, in] on CPU; _compute_hinv_batched does Cholesky on CPU
        hinv = self._compute_hinv_batched(hessian)  # [CH, in, in] on CPU
        if self._hinv_fallback is not None:
            layer_name = kwargs.get("layer_name", "unknown")
            logger.warning(
                f"[GPTQ] Hessian inverse fallback '{self._hinv_fallback}' triggered for "
                f"layer={layer_name}, CH={CH}, infeatures={in_}, "
                f"expert_range=[{self._expert_start_idx}..{self._expert_start_idx + CH - 1}], "
                f"nsamples={self.nsamples.tolist()}"
            )

        losses = torch.zeros_like(w)
        q = torch.zeros_like(w)

        for i1 in range(0, in_, self.blocksize):
            i2 = min(i1 + self.blocksize, in_)
            count = i2 - i1
            w1 = w[:, :, i1:i2].clone()       # [CH, out, count]
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            # Move only the diagonal block [CH, count, count] to GPU
            hinv1 = hinv[:, i1:i2, i1:i2].to(self.dev)  # [CH, count, count]

            for i in range(count):
                col = i1 + i
                w_col = w1[:, :, i]            # [CH, out]
                d = hinv1[:, i, i]             # [CH]

                if self.groupsize != -1:
                    group_idx = col // self.groupsize
                    scale = self.scales[group_idx]  # [CH, out, 1]
                    zero = self.zeros[group_idx]
                else:
                    scale = self.scale
                    zero = self.zero

                q_col = quantize_with_scale_zero(
                    w_col.unsqueeze(-1), scale, zero, bits=self.wbits
                ).squeeze(-1)  # [CH, out]
                q1[:, :, i] = q_col
                losses1[:, :, i] = (w_col - q_col) ** 2 / (d.unsqueeze(-1) ** 2)

                e1 = (w_col - q_col) / d.unsqueeze(-1)  # [CH, out]
                # Batched rank-1 update: w1[:, :, i:] -= e1 @ hinv1[:, i, i:]
                w1[:, :, i:] -= torch.bmm(
                    e1.unsqueeze(2),               # [CH, out, 1]
                    hinv1[:, i, i:].unsqueeze(1),  # [CH, 1, count-i]
                )
                err1[:, :, i] = e1

            q[:, :, i1:i2] = q1
            losses[:, :, i1:i2] = losses1 / 2
            # Move only the off-diagonal block [CH, count, in-i2] to GPU
            hinv_cross = hinv[:, i1:i2, i2:].to(self.dev)  # [CH, count, in-i2]
            w[:, :, i2:] -= torch.bmm(err1, hinv_cross)
            del hinv1, hinv_cross

        # Write results back to individual slice linears
        for local_e in range(CH):
            self._slice_linears[local_e].weight.data = q[local_e].to(
                self._slice_linears[local_e].weight.data.dtype
            )

        # Prepare return values (per-expert)
        if self.groupsize != -1:
            final_scale = torch.cat(self.scales, dim=2)  # [CH, out, num_groups]
            final_zero = torch.cat(self.zeros, dim=2)
            g_idx = (torch.arange(in_, device=w.device) // self.groupsize).to(torch.int32)
        else:
            final_scale = self.scale
            final_zero = self.zero
            g_idx = torch.zeros(in_, dtype=torch.int32, device=w.device)

        avg_loss = float(torch.sum(losses).item() / max(float(self.nsamples.sum().item()), 1))
        # Compute norm_loss on CPU to avoid ~384 MB GPU allocation
        # at the end of fasterquant when GPU memory is tight.
        # Release the cached GPU weight batch first; reconstruct the reference
        # weights directly from the (possibly CPU-offloaded) 3D Parameter.
        self._w_batch = None
        ref_weight = torch.stack(
            [sl._weight_3d.data[sl._expert_idx] for sl in self._slice_linears], dim=0
        )
        norm_loss = float(
            torch.norm(
                q.cpu().float() - ref_weight.cpu().float()
            ).item()
        )
        self.last_metrics = {
            "rows": out, "columns": in_,
            "avg_loss": avg_loss, "norm_loss": norm_loss,
            "nsamples": float(self.nsamples.sum().item()),
            "fallback": self._hinv_fallback,
        }
        self.free()
        # Release cached GPU memory after each batched fasterquant to
        # prevent CUDA allocator fragmentation across 32+ expert batches.
        # Without this, after 8 batches of gate_up_proj (each ~3.7 GB
        # alloc/free), the allocator can't find a contiguous 1.2 GB block
        # for the 9th batch's Hessian on a 32 GB GPU.
        bh.empty_cache()
        return final_scale.cpu(), final_zero.cpu(), g_idx.cpu()

    def write_back(self) -> None:
        for sl in self._slice_linears:
            sl.write_back()

    def free(self) -> None:
        self.H = None
        self._w_batch = None


class GPTQModule(BaseHessianModule):
    """Per-linear GPTQ optimization module."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        wbits: int = 4,
        groupsize: int = 128,
        blocksize: int = 128,
        actorder: bool = True,
        static_groups: bool = True,
        sym: bool = True,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        hessian_device=None,
    ):
        super().__init__(
            layer=layer,
            percdamp=percdamp,
            preproc_hessian=preproc_hessian,
            hessian_device=hessian_device,
        )
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.blocksize = int(blocksize)
        self.actorder = bool(actorder)
        self.static_groups = bool(static_groups)
        self.sym = bool(sym)

        self.scales: List[torch.Tensor] = []
        self.zeros: List[torch.Tensor] = []
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.last_metrics: Dict[str, float] = {}
        self._hinv_fallback: Optional[str] = None

    def fasterquant(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = kwargs
        w_orig = self.layer.weight.data.float().clone().to(self.dev)
        w = self.layer.weight.data.clone().float().to(self.dev)
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if _is_transformers_conv1d(self.layer):
            w = w.t()

        # Apply dead-column masking from preproc without mutating the original weight.
        if self.preproc_hessian and self._dead_mask is not None:
            w = w.masked_fill(self._dead_mask.to(w.device).unsqueeze(0), 0)

        if self.actorder and not self.static_groups and self.groupsize != -1:
            logger.warning(
                "[GPTQ] actorder=True with static_groups=False is unsupported; "
                "forcing static_groups=True."
            )
            self.static_groups = True

        if self.static_groups and self.groupsize != -1:
            for i in range(0, self.columns, self.groupsize):
                scale, zero = compute_scales_with_zero(
                    w[:, i : i + self.groupsize], bits=self.wbits, sym=self.sym
                )
                self.scales.append(scale)
                self.zeros.append(zero)

        if self.groupsize == -1:
            self.scale, self.zero = compute_scales_with_zero(
                w, bits=self.wbits, sym=self.sym
            )

        hessian = self.H.to(self.dev)
        if self.actorder:
            perm = torch.argsort(torch.diag(hessian), descending=True)
            w = w[:, perm]
            hessian = hessian[perm][:, perm]
            invperm = (
                torch.argsort(perm.float()) if bh.has_npu else torch.argsort(perm)
            )
        else:
            perm = torch.arange(self.columns, device=w.device)
            invperm = perm

        hinv = self.compute_hinv(hessian)
        if self._hinv_fallback is not None:
            layer_name = kwargs.get("layer_name", "unknown")
            logger.warning(
                f"[GPTQ] Hessian inverse fallback '{self._hinv_fallback}' triggered for "
                f"layer={layer_name}, rows={self.rows}, columns={self.columns}, "
                f"nsamples={self.nsamples}"
            )
        losses = torch.zeros_like(w)
        q = torch.zeros_like(w)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            w1 = w[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w_col = w1[:, i]
                d = hinv1[i, i]

                if self.groupsize != -1:
                    if self.static_groups:
                        original_idx = int(perm[col].item())
                        group_idx = original_idx // self.groupsize
                        scale = self.scales[group_idx]
                        zero = self.zeros[group_idx]
                    else:
                        if col % self.groupsize == 0:
                            scale, zero = compute_scales_with_zero(
                                w[:, col : col + self.groupsize],
                                bits=self.wbits,
                                sym=self.sym,
                            )
                            self.scales.append(scale)
                            self.zeros.append(zero)
                        else:
                            group_idx = col // self.groupsize
                            scale = self.scales[group_idx]
                            zero = self.zeros[group_idx]
                else:
                    assert self.scale is not None and self.zero is not None
                    scale = self.scale
                    zero = self.zero

                q_col = quantize_with_scale_zero(
                    w_col.unsqueeze(1), scale, zero, bits=self.wbits
                ).flatten()
                q1[:, i] = q_col
                losses1[:, i] = (w_col - q_col) ** 2 / d**2

                e1 = (w_col - q_col) / d
                w1[:, i:] -= e1.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = e1

            q[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2
            w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

        if self.groupsize == -1:
            g_idx = torch.zeros(self.columns, dtype=torch.int32, device=self.layer.weight.device)
        else:
            if self.static_groups:
                g_idx = (perm // self.groupsize).to(torch.int32)
            else:
                g_idx = (torch.arange(self.columns, device=self.layer.weight.device) // self.groupsize).to(
                    torch.int32
                )

        if self.actorder:
            q = q[:, invperm]
            g_idx = g_idx[invperm]

        if _is_transformers_conv1d(self.layer):
            q = q.t()

        self.layer.weight.data = q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.groupsize != -1:
            final_scale = torch.cat(self.scales, dim=1)
            final_zero = torch.cat(self.zeros, dim=1)
        else:
            assert self.scale is not None and self.zero is not None
            final_scale = self.scale
            final_zero = self.zero

        q_reshaped = q.reshape(self.layer.weight.shape).float()
        avg_loss = float(torch.sum(losses).item() / max(int(self.nsamples), 1))
        norm_loss = float(torch.norm(q_reshaped - w_orig).item())
        self.last_metrics = {
            "rows": self.rows,
            "columns": self.columns,
            "avg_loss": avg_loss,
            "norm_loss": norm_loss,
            "nsamples": float(self.nsamples),
            "fallback": self._hinv_fallback,
        }

        self.postproc()
        self.free()
        return final_scale.cpu(), final_zero.cpu(), g_idx.cpu()


class GPTQQuantLinear(nn.Module):
    """Packing helper with v1-compatible tensor layouts."""

    def __init__(
        self,
        bits: int,
        group_size: int,
        infeatures: int,
        outfeatures: int,
        bias: bool,
        weight_dtype: torch.dtype,
        backend: str,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2/3/4/8 bits are supported.")

        self.infeatures = int(infeatures)
        self.outfeatures = int(outfeatures)
        self.bits = int(bits)
        self.group_size = self.infeatures if int(group_size) == -1 else int(group_size)
        self.maxq = 2**self.bits - 1
        self._is_ascend_format = backend == "npu"

        if self._is_ascend_format:
            if self.bits != 4:
                raise ValueError("Ascend backend only supports 4-bit GPTQ packing.")
            if self.infeatures % 8 != 0:
                raise ValueError(
                    f"Ascend packing requires infeatures divisible by 8, got {self.infeatures}"
                )
            num_groups = math.ceil(self.infeatures / self.group_size)
            self.register_buffer(
                "weight",
                torch.zeros((self.outfeatures, self.infeatures // 8), dtype=torch.int32),
            )
            self.register_buffer(
                "weight_scale",
                torch.zeros((self.outfeatures, num_groups), dtype=torch.bfloat16),
            )
            self.register_buffer(
                "weight_offset",
                torch.zeros((self.outfeatures, num_groups), dtype=torch.bfloat16),
            )
        else:
            self.register_buffer(
                "qweight",
                torch.zeros((self.infeatures // 32 * self.bits, self.outfeatures), dtype=torch.int32),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (math.ceil(self.infeatures / self.group_size), self.outfeatures // 32 * self.bits),
                    dtype=torch.int32,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(self.infeatures / self.group_size), self.outfeatures),
                    dtype=weight_dtype,
                ),
            )
            self.register_buffer(
                "g_idx",
                torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32),
            )

        if bias:
            self.register_buffer("bias", torch.zeros((self.outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

    def _pack_ascend(
        self,
        linear: SimpleNamespace,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        weight = linear.weight.data.clone()
        outfeatures, infeatures = weight.shape
        device = weight.device
        if g_idx is None:
            g_idx = torch.arange(infeatures, device=device) // self.group_size

        current_scales = scales[:, g_idx]
        current_scale_zeros = (zeros * scales)[:, g_idx]
        signed_offset = 2 ** (self.bits - 1)
        quant = torch.round((weight + current_scale_zeros) / current_scales) - signed_offset
        quant = quant.clamp(-signed_offset, signed_offset - 1).to(torch.int8)
        quant_unsigned = (quant + signed_offset).to(torch.uint8)

        packed = torch.zeros((outfeatures, infeatures // 8), dtype=torch.int32, device=device)
        for i in range(8):
            cols = torch.arange(i, infeatures, 8, device=device)
            packed |= quant_unsigned[:, cols].to(torch.int32) << (self.bits * i)

        self.weight = packed.contiguous().cpu()
        self.weight_scale = scales.to(torch.bfloat16).cpu()
        self.weight_offset = torch.zeros_like(zeros, dtype=torch.bfloat16).cpu()
        if linear.bias is not None and self.bias is not None:
            self.bias = linear.bias.clone().cpu().to(dtype=self.bias.dtype)

    def _pack_gptq(
        self,
        linear: SimpleNamespace,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        weight = linear.weight.data.clone()
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=self.scales.dtype)
        if linear.bias is not None and self.bias is not None:
            self.bias = linear.bias.clone().to(dtype=self.bias.dtype)

        # Vectorized intweight computation: expand per-group scales/zeros to
        # per-column and compute all columns at once (replaces Python for-loop
        # that iterated self.infeatures times, ~100x faster for infeatures=4096).
        # Memory overhead: ~1x weight size for expanded scales (temporary, freed
        # immediately after this block).
        current_scales = self.scales[self.g_idx].t()       # [outfeatures, infeatures]
        current_scale_zeros = scale_zeros[self.g_idx].t()  # [outfeatures, infeatures]
        intweight = torch.round((weight + current_scale_zeros) / current_scales).to(torch.int)
        intweight = intweight.t().contiguous().numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:  # pragma: no cover
                raise NotImplementedError("Only 2/3/4/8 bits are supported.")
        self.qweight = torch.from_numpy(qweight.astype(np.int32))

        zeros = (zeros - 1).numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:  # pragma: no cover
                raise NotImplementedError("Only 2/3/4/8 bits are supported.")
        self.qzeros = torch.from_numpy(qzeros.astype(np.int32))

    def pack(
        self,
        linear: SimpleNamespace,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if self._is_ascend_format:
            self._pack_ascend(linear, scales, zeros, g_idx)
        else:
            self._pack_gptq(linear, scales, zeros, g_idx)


@AlgorithmRegistry.register(
    "GPTQ",
    aliases=["gptq", "GPTQStepwise", "GPTQExample", "gptq_stepwise"],
)
class GPTQAlgorithm(BaseHessianAlgorithm):
    """Chunk-wise GPTQ algorithm."""

    _TAG = "GPTQ"
    _quantized_type_label = "GPTQ"

    def __init__(
        self,
        wbits: int = 4,
        w_bits: Optional[int] = None,
        groupsize: int = 128,
        group_size: Optional[int] = None,
        sym: bool = True,
        blocksize: int = 128,
        actorder: bool = True,
        static_groups: bool = True,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        fake_quant: bool = False,
        max_calib_samples: int = 128,
        save_backend: Optional[str] = None,
        expert_chunk_size: int = 1,
        quantize_mtp: bool = False,
        save_mtp_debug: bool = False,
        **kwargs,
    ):
        if w_bits is not None:
            wbits = int(w_bits)
        if group_size is not None:
            groupsize = int(group_size)
        super().__init__(max_calib_samples=max_calib_samples, quantize_mtp=quantize_mtp, save_mtp_debug=save_mtp_debug, **kwargs)
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.sym = bool(sym)
        self.blocksize = int(blocksize)
        self.actorder = bool(actorder)
        self.static_groups = bool(static_groups)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self.expert_chunk_size = max(int(expert_chunk_size), 1)
        self.fake_quant = bool(fake_quant)
        self._save_backend = save_backend

    @property
    def _ascend_quant_type(self) -> str:
        return "FLOAT" if self.fake_quant else f"W{self.wbits}A16"

    def _log_start_params(self) -> None:
        logger.info(
            f"[{self._TAG}] start: "
            f"wbits={self.wbits}, groupsize={self.groupsize}, "
            f"actorder={self.actorder}, percdamp={self.percdamp}"
        )

    def _pack_quant_linear_tensors(
        self,
        module_name: str,
        linear_module: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        quant_linear = GPTQQuantLinear(
            bits=self.wbits,
            group_size=self.groupsize,
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            bias=linear_module.bias is not None,
            weight_dtype=linear_module.weight.dtype,
            backend=self.target_backend,
        )
        proxy = SimpleNamespace(
            weight=linear_module.weight.detach().cpu(),
            bias=(linear_module.bias.detach().cpu() if linear_module.bias is not None else None),
        )
        quant_linear.pack(proxy, scales.cpu(), zeros.cpu(), g_idx.cpu())

        tensors: Dict[str, torch.Tensor] = {}
        quantized_names: List[str] = []
        if self.target_backend == "npu":
            tensors[f"{module_name}.weight"] = quant_linear.weight.cpu()
            tensors[f"{module_name}.weight_scale"] = quant_linear.weight_scale.cpu()
            tensors[f"{module_name}.weight_offset"] = quant_linear.weight_offset.cpu()
            quantized_names.extend(
                [
                    f"{module_name}.weight",
                    f"{module_name}.weight_scale",
                    f"{module_name}.weight_offset",
                ]
            )
        else:
            tensors[f"{module_name}.qweight"] = quant_linear.qweight.cpu()
            tensors[f"{module_name}.qzeros"] = quant_linear.qzeros.cpu()
            tensors[f"{module_name}.scales"] = quant_linear.scales.cpu()
            tensors[f"{module_name}.g_idx"] = quant_linear.g_idx.cpu()
            quantized_names.extend(
                [
                    f"{module_name}.qweight",
                    f"{module_name}.qzeros",
                    f"{module_name}.scales",
                    f"{module_name}.g_idx",
                ]
            )
        if quant_linear.bias is not None:
            tensors[f"{module_name}.bias"] = quant_linear.bias.cpu()
        return tensors, quantized_names

    def _create_handlers(self, layer_module: nn.Module, targets) -> Dict[str, Any]:
        handlers: Dict[str, Any] = {}
        for target in targets:
            module_rel_name = target[0]
            is_3d = target[5] if len(target) > 5 else False

            if is_3d:
                # 3D fused MoE Parameter
                parts = module_rel_name.split(".")
                param_name = parts[-1]
                experts_module_path = ".".join(parts[:-1])
                experts_module = _get_child_module(layer_module, experts_module_path)
                if experts_module is None:
                    continue
                weight_3d = getattr(experts_module, param_name, None)
                if weight_3d is None or not isinstance(weight_3d, nn.Parameter):
                    continue
                num_experts = weight_3d.shape[0]
                CH = min(self.expert_chunk_size, num_experts)

                if CH > 1:
                    # Batched: create num_experts/CH BatchedGPTQModule handlers
                    for chunk_start in range(0, num_experts, CH):
                        chunk_end = min(chunk_start + CH, num_experts)
                        slice_linears = [
                            _ExpertSliceLinear(weight_3d, e, param_name, experts_module_path)
                            for e in range(chunk_start, chunk_end)
                        ]
                        handler_key = f"{experts_module_path}.{chunk_start}.{param_name}"
                        handlers[handler_key] = BatchedGPTQModule(
                            slice_linears,
                            wbits=self.wbits,
                            groupsize=self.groupsize,
                            blocksize=self.blocksize,
                            sym=self.sym,
                            percdamp=self.percdamp,
                            preproc_hessian=self.preproc_hessian,
                            hessian_device=torch.device("cpu"),
                        )
                else:
                    # Per-expert: create num_experts GPTQModule handlers
                    for e in range(num_experts):
                        expert_rel_name = f"{experts_module_path}.{e}.{param_name}"
                        slice_linear = _ExpertSliceLinear(
                            weight_3d, e, param_name, experts_module_path
                        )
                        handlers[expert_rel_name] = GPTQModule(
                            slice_linear,
                            wbits=self.wbits,
                            groupsize=self.groupsize,
                            blocksize=self.blocksize,
                            actorder=self.actorder,
                            static_groups=self.static_groups,
                            sym=self.sym,
                            percdamp=self.percdamp,
                            preproc_hessian=self.preproc_hessian,
                            hessian_device=torch.device("cpu"),
                        )
            else:
                # Standard 2D Linear
                submodule = _get_child_module(layer_module, module_rel_name)
                if not isinstance(submodule, nn.Linear):
                    continue
                hessian_device = None
                if ".experts." in module_rel_name:
                    hessian_device = torch.device("cpu")
                handlers[module_rel_name] = GPTQModule(
                    submodule,
                    wbits=self.wbits,
                    groupsize=self.groupsize,
                    blocksize=self.blocksize,
                    actorder=self.actorder,
                    static_groups=self.static_groups,
                    sym=self.sym,
                    percdamp=self.percdamp,
                    preproc_hessian=self.preproc_hessian,
                    hessian_device=hessian_device,
                )
        return handlers

    _EXPERT_PACKED_RE = re.compile(r"^(.+)\.experts\.(\d+)\.([^.]+)\.(.+)$")

    # NPU (Ascend) suffixes: output dimension is dim=0 (weight is [out, in//8])
    _NPU_SUFFIXES = {"weight", "weight_scale", "weight_offset"}
    # GPU (GPTQ) suffixes: output dimension is dim=1 (qweight is [in//8, out])
    _GPU_SUFFIXES = {"qweight", "qzeros", "scales"}
    # g_idx is 1D [infeatures], identical for gate_proj and up_proj; do NOT concatenate
    _G_IDX_SUFFIX = "g_idx"

    def _get_cat_dim(self, suffix: str) -> int:
        """Determine cat dimension for gate+up fusion based on tensor suffix.

        NPU format: weight [out, in//8] -> cat dim=0 (output first)
        GPU format: qweight [in//8, out] -> cat dim=1 (output last)
        g_idx: 1D [infeatures], identical for all components -> no cat (return -1)
        """
        if suffix == self._G_IDX_SUFFIX:
            return -1  # Special: take first copy, don't concatenate
        if suffix in self._GPU_SUFFIXES:
            return 1  # GPU GPTQ: output is dim 1
        return 0  # NPU Ascend: output is dim 0

    def _split_moe_gate_up_proj(
        self, layer, quantized_tensor_names: set[str]
    ) -> tuple[set[str], int]:
        """Split per-expert gate_up_proj into separate gate_proj + up_proj.

        Produces per-expert 2D naming (experts.0.gate_proj.weight) instead of
        fused naming (experts.gate_up_proj.weight), compatible with vLLM's
        expert_params_mapping for both W8A8 and W4A16 on Ascend NPU.

        down_proj is already per-expert 2D and needs no splitting.
        """
        fusion_map = getattr(self._model_obj, "moe_expert_fusion_map", {})
        if not fusion_map:
            return quantized_tensor_names, 0

        # Build component-to-fused mapping
        component_to_fused: dict[str, str] = {}
        for fused_name, (components, _) in fusion_map.items():
            for comp in components:
                component_to_fused[comp] = fused_name
            component_to_fused[fused_name] = fused_name

        # Find per-expert fused tensors (e.g. gate_up_proj) that need splitting
        to_split: dict[tuple, tuple[torch.Tensor, list[str]]] = {}
        to_remove: list[str] = []

        for rel_name, tensor in layer.tensors.items():
            match = self._EXPERT_PACKED_RE.match(rel_name)
            if not match:
                continue
            prefix = match.group(1)
            expert_idx = int(match.group(2))
            component = match.group(3)
            suffix = match.group(4)

            if component not in component_to_fused:
                continue
            fused_name = component_to_fused[component]
            fusion_components, _ = fusion_map[fused_name]
            # Only split multi-component fusions (gate_up_proj = gate + up)
            if len(fusion_components) <= 1:
                continue

            to_split[(prefix, expert_idx, fused_name, suffix)] = (
                tensor,
                fusion_components,
            )
            to_remove.append(rel_name)

        if not to_split:
            return quantized_tensor_names, 0

        new_tensors: dict[str, torch.Tensor] = {}
        new_quantized_names: set[str] = set()
        split_count = 0

        for (prefix, expert_idx, fused_name, suffix), (
            tensor,
            components,
        ) in to_split.items():
            cat_dim = self._get_cat_dim(suffix)
            if cat_dim == -1:
                # g_idx: 1D, identical for all components - just copy
                for comp in components:
                    new_name = (
                        f"{prefix}.experts.{expert_idx}.{comp}.{suffix}"
                    )
                    new_tensors[new_name] = tensor.clone()
                    new_quantized_names.add(f"{layer.name}.{new_name}")
                    split_count += 1
            else:
                # Split along cat_dim (dim=0 for NPU, dim=1 for GPU)
                chunks = torch.chunk(tensor, len(components), dim=cat_dim)
                for i, comp in enumerate(components):
                    new_name = (
                        f"{prefix}.experts.{expert_idx}.{comp}.{suffix}"
                    )
                    new_tensors[new_name] = chunks[i].clone()
                    new_quantized_names.add(f"{layer.name}.{new_name}")
                    split_count += 1

        # Remove old fused per-expert names
        for rel_name in to_remove:
            layer.tensors.pop(rel_name, None)
            quantized_tensor_names.discard(f"{layer.name}.{rel_name}")

        # Add new split names
        for rel_name, tensor in new_tensors.items():
            layer.tensors[rel_name] = tensor
            quantized_tensor_names.add(f"{layer.name}.{rel_name}")

        return quantized_tensor_names, split_count

    def _refuse_moe_expert_tensors(
        self, layer, quantized_tensor_names: set[str]
    ) -> tuple[set[str], int]:
        """Re-fuse per-expert packed tensors into 3D format for vLLM compatibility.

        Handles two naming patterns:
        - 3D approach: experts.0.gate_up_proj.weight -> experts.gate_up_proj.weight (stack only)
        - Expanded approach: experts.0.gate_proj.weight + experts.0.up_proj.weight -> experts.gate_up_proj.weight (cat + stack)

        Automatically selects correct cat dimension based on tensor suffix:
        - NPU (weight/weight_scale/weight_offset): cat dim=0 (output is dim 0)
        - GPU (qweight/qzeros/scales): cat dim=1 (output is dim 1)
        - g_idx: no cat (identical for gate_proj and up_proj)
        """
        fusion_map = getattr(self._model_obj, "moe_expert_fusion_map", {})
        if not fusion_map:
            return quantized_tensor_names, 0

        # Map component names AND fused names to fused names
        component_to_fused: Dict[str, str] = {}
        for fused_name, (components, _op) in fusion_map.items():
            for comp in components:
                component_to_fused[comp] = fused_name
            # Also map fused name to itself (3D approach uses fused names directly)
            component_to_fused[fused_name] = fused_name

        collected: Dict[tuple, Dict[int, Dict[str, torch.Tensor]]] = {}
        to_remove: List[str] = []

        for rel_name, tensor in layer.tensors.items():
            match = self._EXPERT_PACKED_RE.match(rel_name)
            if not match:
                continue
            prefix = match.group(1)
            expert_idx = int(match.group(2))
            component = match.group(3)
            suffix = match.group(4)
            if component not in component_to_fused:
                continue
            fused_name = component_to_fused[component]
            key = (prefix, fused_name, suffix)
            collected.setdefault(key, {}).setdefault(expert_idx, {})[component] = tensor
            to_remove.append(rel_name)

        if not collected:
            return quantized_tensor_names, 0

        new_tensors: Dict[str, torch.Tensor] = {}
        new_quantized_names: set[str] = set()

        for (prefix, fused_name, suffix), experts_dict in collected.items():
            fusion_components, _ = fusion_map[fused_name]
            cat_dim = self._get_cat_dim(suffix)
            expert_slices: List[torch.Tensor] = []
            for e in sorted(experts_dict.keys()):
                comp_dict = experts_dict[e]
                if len(comp_dict) > 1:
                    # Multiple components (expanded approach): cat in fusion_map order
                    parts = [comp_dict[c] for c in fusion_components if c in comp_dict]
                    if parts:
                        if cat_dim == -1:
                            # g_idx: identical for all components, take first copy
                            expert_slices.append(parts[0])
                        else:
                            expert_slices.append(torch.cat(parts, dim=cat_dim))
                else:
                    # Single component (3D approach or down_proj): just take it
                    for t in comp_dict.values():
                        expert_slices.append(t)
                        break
            if expert_slices:
                fused_tensor = torch.stack(expert_slices, dim=0)
                fused_rel_name = f"{prefix}.experts.{fused_name}.{suffix}"
                new_tensors[fused_rel_name] = fused_tensor
                new_quantized_names.add(f"{layer.name}.{fused_rel_name}")

        for rel_name in to_remove:
            layer.tensors.pop(rel_name, None)
            quantized_tensor_names.discard(f"{layer.name}.{rel_name}")

        for rel_name, tensor in new_tensors.items():
            layer.tensors[rel_name] = tensor
            quantized_tensor_names.add(f"{layer.name}.{rel_name}")

        return quantized_tensor_names, len(new_tensors)

    def _process_layer_handlers(self, layer, targets, handlers, chunk) -> tuple[set[str], int]:
        quantized_tensor_names: set[str] = set()
        quantized_weights = 0
        quant_results = []

        # Process 3D targets first, then 2D targets.
        # 3D targets hold 19.3 GB of Parameters; after each is done the
        # Parameter is moved to CPU, freeing VRAM for the remaining targets.
        targets_3d = [t for t in targets if len(t) > 5 and t[5]]
        targets_2d = [t for t in targets if not (len(t) > 5 and t[5])]

        for target in targets_3d + targets_2d:
            module_rel_name = target[0]
            rel_weight_name = target[1]
            rel_bias_name = target[2]
            weight_tensor = target[3]
            is_3d = target[5] if len(target) > 5 else False

            if is_3d:
                # 3D fused MoE Parameter: iterate over chunk handlers
                parts = module_rel_name.split(".")
                param_name = parts[-1]
                experts_module_path = ".".join(parts[:-1])
                num_experts = weight_tensor.shape[0]
                weight_dtype = weight_tensor.dtype
                CH = min(self.expert_chunk_size, num_experts)
                found_handler = False

                # Release the original 3D tensor from layer.tensors AND from the
                # targets list early. _collect_statistics already moved w3d.data
                # to a new CPU tensor; fasterquant only uses w3d via _slice_linears,
                # not this original.  Clearing both references (layer.tensors dict
                # entry + targets tuple entry) frees ~19 GB (GLM-5) during the
                # fasterquant loop.  Tuples are immutable, so replace the entry in
                # the original targets list with a copy whose [3] is None.
                layer.tensors.pop(rel_weight_name, None)
                for j in range(len(targets)):
                    if targets[j] is target:
                        targets[j] = target[:3] + (None,) + target[4:]
                        break
                weight_tensor = None  # release local reference too

                # 3D Parameters were already moved to CPU in _collect_statistics.
                # Log device memory at the start of 3D processing for diagnostics.
                if bh.has_cuda:
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        f"[{self._TAG}] 3D quant start: {module_rel_name}, "
                        f"GPU allocated={mem_alloc:.2f} GB, reserved={mem_reserved:.2f} GB"
                    )

                for chunk_start in range(0, num_experts, CH):
                    handler_key = f"{experts_module_path}.{chunk_start}.{param_name}"
                    handler = handlers.get(handler_key)
                    if handler is None:
                        continue
                    found_handler = True

                    is_batched = getattr(handler, "_is_batched_expert", False)
                    actual_ch = handler._CH if is_batched else 1

                    all_scales, all_zeros, all_g_idx = handler.fasterquant(
                        layer_name=f"{layer.name}.{handler_key}"
                    )
                    self._record_fallback(handler, f"{layer.name}.{handler_key}")
                    if is_batched:
                        handler.write_back()
                        # Move quantized weights to CPU immediately after write_back.
                        # fasterquant sets sl.weight.data to GPU tensor q[local_e].
                        # Without this, 256 experts × 25-50 MB = 6-13 GB of "zombie"
                        # GPU tensors accumulate across batches, causing OOM.
                        for sl in handler._slice_linears:
                            sl.weight.data = sl.weight.data.to("cpu")
                    else:
                        handler.layer.write_back()
                        handler.layer.weight.data = handler.layer.weight.data.to("cpu")

                    metrics = getattr(handler, "last_metrics", {})
                    if metrics:
                        logger.info(
                            f"[{self._TAG}] {layer.name}.{handler_key:<44s} | "
                            f"chunk={actual_ch} | "
                            f"shape=[{int(metrics.get('rows', 0)):>5},{int(metrics.get('columns', 0)):>5}] | "
                            f"avg_loss={float(metrics.get('avg_loss', 0.0)):<12.6f} | "
                            f"norm_loss={float(metrics.get('norm_loss', 0.0)):<12.6f}"
                        )

                    for local_e in range(actual_ch):
                        e = chunk_start + local_e
                        expert_rel_name = f"{experts_module_path}.{e}.{param_name}"

                        if is_batched:
                            slice_linear = handler._slice_linears[local_e]
                            expert_scales = all_scales[local_e]
                            expert_zeros = all_zeros[local_e]
                        else:
                            slice_linear = handler.layer
                            expert_scales = all_scales
                            expert_zeros = all_zeros

                        if self.fake_quant:
                            expert_weight_name = f"{expert_rel_name}.weight"
                            layer.tensors[expert_weight_name] = (
                                slice_linear.weight.detach().to(weight_dtype).cpu()
                            )
                            quantized_tensor_names.add(f"{layer.name}.{expert_weight_name}")
                        else:
                            packed_tensors, packed_quant_names = self._pack_quant_linear_tensors(
                                module_name=expert_rel_name,
                                linear_module=slice_linear,
                                scales=expert_scales,
                                zeros=expert_zeros,
                                g_idx=all_g_idx,
                            )
                            for rel_name, t in packed_tensors.items():
                                layer.tensors[rel_name] = t
                            for rel_quant_name in packed_quant_names:
                                quantized_tensor_names.add(f"{layer.name}.{rel_quant_name}")
                        quantized_weights += 1

                # Original 3D weight was already popped from layer.tensors
                # before the fasterquant loop to free ~19 GB during quantization.
                if found_handler:
                    # Clear CUDA cache to defragment after 64+ fasterquant batches.
                    bh.empty_cache()
            else:
                # Standard 2D Linear
                handler = handlers.get(module_rel_name)
                if handler is None:
                    continue
                scales, zeros, g_idx = handler.fasterquant(
                    layer_name=f"{layer.name}.{module_rel_name}"
                )
                self._record_fallback(handler, f"{layer.name}.{module_rel_name}")
                metrics = getattr(handler, "last_metrics", {})
                if metrics:
                    full_name = f"{layer.name}.{module_rel_name}"
                    logger.info(
                        f"[{self._TAG}] {full_name:<50s} | "
                        f"shape=[{int(metrics.get('rows', 0)):>5},{int(metrics.get('columns', 0)):>5}] | "
                        f"avg_loss={float(metrics.get('avg_loss', 0.0)):<12.6f} | "
                        f"norm_loss={float(metrics.get('norm_loss', 0.0)):<12.6f}"
                    )
                quant_results.append(
                    (module_rel_name, rel_weight_name, rel_bias_name, weight_tensor, scales, zeros, g_idx, handler)
                )

        # Pack 2D results
        pack_iter = tqdm(
            quant_results,
            total=len(quant_results),
            desc=f"{self._TAG.lower()} pack c{chunk.chunk_index} {layer.name}",
            leave=True,
            disable=len(quant_results) <= 1,
        )
        for module_rel_name, rel_weight_name, rel_bias_name, weight_tensor, scales, zeros, g_idx, handler in pack_iter:
            if self.fake_quant:
                layer.tensors[rel_weight_name] = (
                    handler.layer.weight.detach().to(weight_tensor.dtype).cpu()
                )
                quantized_tensor_names.add(f"{layer.name}.{rel_weight_name}")
            else:
                packed_tensors, packed_quant_names = self._pack_quant_linear_tensors(
                    module_name=module_rel_name,
                    linear_module=handler.layer,
                    scales=scales,
                    zeros=zeros,
                    g_idx=g_idx,
                )
                layer.tensors.pop(rel_weight_name, None)
                layer.tensors.pop(rel_bias_name, None)
                for rel_name, tensor in packed_tensors.items():
                    layer.tensors[rel_name] = tensor
                for rel_quant_name in packed_quant_names:
                    quantized_tensor_names.add(f"{layer.name}.{rel_quant_name}")
            quantized_weights += 1

        # Split per-expert gate_up_proj into gate_proj + up_proj for vLLM
        # expert_params_mapping compatibility (per-expert 2D format)
        quantized_tensor_names, _ = self._split_moe_gate_up_proj(layer, quantized_tensor_names)
        return quantized_tensor_names, quantized_weights

    def _update_quantization_metadata(self) -> None:
        if self._model_config is None:
            return

        if self.target_backend == "npu":
            self._model_config.ascend_quant_config = {
                "model_quant_type": self._ascend_quant_type,
                "group_size": self.groupsize,
                "quant_layer_types": ["GPTQQuantLinear"],
                "include_g_idx": True,
                "has_offset": True,
            }
            if hasattr(self._model_config, "quantization_config"):
                try:
                    delattr(self._model_config, "quantization_config")
                except Exception:
                    pass
        else:
            self._model_config.quantization_config = {
                "bits": self.wbits,
                "group_size": self.groupsize,
                "sym": self.sym,
                "desc_act": self.actorder,
                "static_groups": self.static_groups,
                "quant_method": "gptq",
                "checkpoint_format": "gptq",
                "true_sequential": True,
            }

        self._mark_model_quantized()
        logger.info(f"[{self._TAG}] model quantization metadata updated")

    # === MTP eh_proj support ===

    def _create_eh_proj_handler(self, mtp_module) -> Optional[Any]:
        """Create a GPTQModule handler for eh_proj."""
        eh_proj = mtp_module.eh_proj
        if not isinstance(eh_proj, nn.Linear):
            return None
        return GPTQModule(
            eh_proj,
            wbits=self.wbits,
            groupsize=self.groupsize,
            blocksize=self.blocksize,
            actorder=self.actorder,
            static_groups=self.static_groups,
            sym=self.sym,
            percdamp=self.percdamp,
            preproc_hessian=self.preproc_hessian,
        )

    def _pack_eh_proj(self, layer, mtp_module, eh_handler) -> tuple[set[str], int]:
        """Quantize and pack eh_proj weights."""
        scales, zeros, g_idx = eh_handler.fasterquant(
            layer_name=f"{layer.name}.eh_proj"
        )
        self._record_fallback(eh_handler, f"{layer.name}.eh_proj")
        metrics = getattr(eh_handler, "last_metrics", {})
        if metrics:
            logger.info(
                f"[{self._TAG}] {layer.name}.eh_proj{'':<42s} | "
                f"shape=[{int(metrics.get('rows', 0)):>5},{int(metrics.get('columns', 0)):>5}] | "
                f"avg_loss={float(metrics.get('avg_loss', 0.0)):<12.6f} | "
                f"norm_loss={float(metrics.get('norm_loss', 0.0)):<12.6f}"
            )

        if self.fake_quant:
            layer.tensors["eh_proj.weight"] = eh_handler.layer.weight.detach().cpu()
            return {f"{layer.name}.eh_proj.weight"}, 1

        packed_tensors, packed_names = self._pack_quant_linear_tensors(
            module_name="eh_proj",
            linear_module=eh_handler.layer,
            scales=scales,
            zeros=zeros,
            g_idx=g_idx,
        )
        # Remove original eh_proj weight
        layer.tensors.pop("eh_proj.weight", None)
        for rel_name, tensor in packed_tensors.items():
            layer.tensors[rel_name] = tensor
        quantized_names = {f"{layer.name}.{n}" for n in packed_names}
        return quantized_names, 1
