"""Chunk-wise GPTQ algorithm for compressor task.

Design goal:
- Strict chunk lifecycle: calibrate -> quantize -> pack inside each process_chunk call.
- No full model load requirement.
- Keep GPTQ core math/packing behavior aligned with v1 implementation.
"""

from __future__ import annotations

import fnmatch
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
        rtn: bool = False,
        ssz: bool = False,
        ssz_step: int = 10,
        clip_search: Optional[Dict] = None,
    ):
        self._slice_linears = slice_linears
        self._CH = len(slice_linears)
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.blocksize = int(blocksize)
        self.sym = bool(sym)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        # E8: RTN mode for batched expert slices — skip Hessian compensation.
        self.rtn = bool(rtn)
        # SSZ per-channel scale refinement (rtn base, ref: msmodelslim ssz.py)
        self.ssz = bool(ssz)
        self.ssz_step = int(ssz_step)
        self.clip_cfg = dict(clip_search) if clip_search else None
        self.omni_steps = int(clip_search.get("omni_steps", 0)) if clip_search else 0
        if self.omni_steps > 0:
            logger.warning(
                "[GPTQ] omni refine is only wired for the per-expert 2D path "
                "(expert_chunk_size=1); ignoring in BatchedGPTQModule"
            )
        self._hdev = hessian_device if hessian_device is not None else slice_linears[0].weight.device

        w0 = slice_linears[0].weight.data
        self.dev = w0.device
        self.outfeatures = w0.shape[0]
        self.infeatures = w0.shape[1]

        # Hessian: accumulated on CPU to avoid GPU OOM with 256 experts.
        # Each batch's contrib (X@X.T) is computed on GPU and immediately
        # transferred to CPU via .to(self._hdev). This is the same pattern
        # as BaseHessianModule.add_batch.
        # E8 rtn mode: Hessian is never accumulated nor consumed — skip the
        # [CH, in, in] allocation entirely.
        if self.rtn:
            self.H = None
        else:
            self.H = torch.zeros(self._CH, self.infeatures, self.infeatures, device=self._hdev)
        self.nsamples = torch.zeros(self._CH, device=self._hdev)
        self._x_abs_sum = torch.zeros(self._CH, self.infeatures, device=self._hdev)  # [CH, in] for scale search
        self._x_abs_max = torch.zeros(self._CH, self.infeatures, device=self._hdev)  # [CH, in] abs-max (AWQ alpha-search stat)
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

    @property
    def x_abs_mean(self) -> torch.Tensor:
        """Per-input-channel activation abs-mean [CH, in] for scale search."""
        ns = self.nsamples.clamp(min=1).to(self._hdev).view(-1, 1)  # [CH, 1]
        return self._x_abs_sum / ns  # [CH, in]

    @property
    def x_abs_max(self) -> torch.Tensor:
        """Per-input-channel activation abs-max [CH, in] (AWQ alpha-search stat)."""
        return self._x_abs_max

    def add_batch_for_expert(self, inp: torch.Tensor, local_idx: int) -> None:
        """Accumulate Hessian contribution on CPU.

        Computes X@X.T on GPU, then immediately transfers to CPU to avoid
        the 38 GB GPU memory that 32×BatchedGPTQModule GPU Hessian accumulators would require.
        PCIe transfer overhead is acceptable for 256-expert MoE layers.
        """
        if self.rtn:
            # E8: Hessian is never consumed in rtn mode — keep activation
            # abs-sum (AWQ scale search) and nsamples (metrics) only.
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            if len(inp.shape) == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t().to(self.dev, dtype=torch.float32)
            batch = inp.shape[1]
            self._x_abs_sum[local_idx] += inp.abs().sum(dim=1).to(self._hdev)
            self._x_abs_max[local_idx] = torch.maximum(
                self._x_abs_max[local_idx], inp.abs().amax(dim=1).to(self._hdev)
            )
            self.nsamples[local_idx] += batch
            self.H = None  # Hessian intentionally unused in rtn mode
            return
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().to(self.dev, dtype=torch.float32)  # [in, batch]
        batch = inp.shape[1]
        # Accumulate per-channel activation abs-mean for scale search (AWQ)
        self._x_abs_sum[local_idx] += inp.abs().sum(dim=1).to(self._hdev)
        self._x_abs_max[local_idx] = torch.maximum(
            self._x_abs_max[local_idx], inp.abs().amax(dim=1).to(self._hdev)
        )
        contrib = inp.matmul(inp.t())  # [in, in] on GPU
        self.H[local_idx] += contrib.to(self._hdev)  # GPU -> CPU transfer
        self.nsamples[local_idx] += batch

    def preproc(self) -> None:
        if self.rtn:
            self.preproc_done = True
            return
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

    def _rtn_quantize(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """E8 expert RTN path (batched): round-to-nearest, no Hessian compensation.

        Same static per-group scale computation and return conventions as
        fasterquant, so downstream packing is unchanged.
        """
        CH, out, in_ = self._CH, self.outfeatures, self.infeatures
        layer_name = kwargs.get("layer_name", "unknown")
        logger.info(
            f"[GPTQ] rtn expert quant: layer={layer_name}, CH={CH}, "
            f"shape=[{out},{in_}], experts=[{self._expert_start_idx}.."
            f"{self._expert_start_idx + CH - 1}]"
        )
        if self.ssz:
            logger.info(
                f"[GPTQ] ssz scale refinement enabled (step={self.ssz_step}) "
                f"for layer={layer_name}, CH={CH}"
            )
        w = self._build_weight_batch().float().clone()  # [CH, out, in]

        q = torch.zeros_like(w)
        if self.groupsize != -1:
            if self.ssz:
                logger.warning(
                    "[GPTQ] ssz scale refinement applies to per-channel "
                    "(groupsize=-1) only; skipping for grouped path"
                )
            self.scales = []
            self.zeros = []
            for i in range(0, in_, self.groupsize):
                chunk = w[:, :, i : i + self.groupsize]
                s, z = compute_scales_with_zero(
                    chunk.reshape(CH * out, self.groupsize), self.wbits, self.sym
                )
                self.scales.append(s.reshape(CH, out, 1))
                self.zeros.append(z.reshape(CH, out, 1))
            for i1 in range(0, in_, self.groupsize):
                i2 = min(i1 + self.groupsize, in_)
                g = i1 // self.groupsize
                q[:, :, i1:i2] = quantize_with_scale_zero(
                    w[:, :, i1:i2], self.scales[g], self.zeros[g], bits=self.wbits
                )
            final_scale = torch.cat(self.scales, dim=2)  # [CH, out, num_groups]
            final_zero = torch.cat(self.zeros, dim=2)
            g_idx = (torch.arange(in_, device=w.device) // self.groupsize).to(torch.int32)
        else:
            # per-channel (groupsize == -1): one scale/zero per (CH, out) row
            s, z = compute_scales_with_zero(
                w.reshape(CH * out, in_), bits=self.wbits, sym=self.sym
            )
            if self.ssz:
                from ..ssz import ssz_calculate_scale

                s = ssz_calculate_scale(
                    w.reshape(CH * out, in_), s, bits=self.wbits, step=self.ssz_step
                )
            if self.clip_cfg:
                from ..ssz import clip_search_scale

                s = clip_search_scale(
                    w.reshape(CH * out, in_),
                    s,
                    bits=self.wbits,
                    lo=float(self.clip_cfg.get("lo", 0.65)),
                    hi=float(self.clip_cfg.get("hi", 1.0)),
                    steps=int(self.clip_cfg.get("steps", 20)),
                    with_ssz=bool(self.clip_cfg.get("with_ssz", False)),
                    ssz_step=self.ssz_step,
                )
            self.scale = s.reshape(CH, out, 1)
            self.zero = z.reshape(CH, out, 1)
            q = quantize_with_scale_zero(w, self.scale, self.zero, bits=self.wbits)
            final_scale = self.scale
            final_zero = self.zero
            g_idx = torch.zeros(in_, dtype=torch.int32, device=w.device)

        # Write results back to individual slice linears
        for local_e in range(CH):
            self._slice_linears[local_e].weight.data = q[local_e].to(
                self._slice_linears[local_e].weight.data.dtype
            )

        self._w_batch = None
        ref_weight = torch.stack(
            [sl._weight_3d.data[sl._expert_idx] for sl in self._slice_linears], dim=0
        )
        norm_loss = float(torch.norm(q.cpu().float() - ref_weight.cpu().float()).item())
        avg_loss = float(torch.mean((q - w) ** 2).item())
        self.last_metrics = {
            "rows": out,
            "columns": in_,
            "avg_loss": avg_loss,
            "norm_loss": norm_loss,
            "nsamples": float(self.nsamples.sum().item()),
            "fallback": None,
        }
        self.free()
        bh.empty_cache()
        return final_scale.cpu(), final_zero.cpu(), g_idx.cpu()

    def fasterquant(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rtn:
            return self._rtn_quantize(**kwargs)
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
        self._x_abs_sum = None
        self._x_abs_max = None


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
        rtn: bool = False,
        ssz: bool = False,
        ssz_step: int = 10,
        clip_search: Optional[Dict] = None,
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
        # E8: RTN mode for expert slices — skip Hessian error compensation.
        self.rtn = bool(rtn)
        # SSZ per-channel scale refinement (rtn base, ref: msmodelslim ssz.py)
        self.ssz = bool(ssz)
        self.ssz_step = int(ssz_step)
        self.clip_cfg = dict(clip_search) if clip_search else None
        if self.rtn:
            self.H = None  # Hessian never accumulated/consumed in rtn mode

        # OmniQuant 式激活感知 scale 精修（§6.9.2#3）：omni_steps>0 时 rtn 分支追加
        # 捕获专家输入激活（低秩 X 形式），量化期对 clip+SSZ 的 scale 做梯度精修。
        self.omni_steps = int(self.clip_cfg.get("omni_steps", 0)) if self.clip_cfg else 0
        self.omni_lr = float(self.clip_cfg.get("omni_lr", 0.02)) if self.clip_cfg else 0.02
        self._omni_x_chunks: Optional[List[torch.Tensor]] = [] if self.omni_steps > 0 else None

        self.scales: List[torch.Tensor] = []
        self.zeros: List[torch.Tensor] = []
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.last_metrics: Dict[str, float] = {}
        self._hinv_fallback: Optional[str] = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        if self.rtn:
            # E8: Hessian is never consumed in rtn mode — skip the [in,batch] @
            # [batch,in] contribution and keep only activation stats (AWQ scale
            # search) and nsamples (metrics). Mirrors BaseHessianModule.add_batch
            # without the H update; expert slices are plain Linear (no conv).
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t().to(self.dev, dtype=torch.float32)
            batch = inp.shape[1]
            abs_sum = inp.abs().sum(dim=1).to(self._hdev)
            abs_max = inp.abs().amax(dim=1).to(self._hdev)
            if self._x_abs_sum is None:
                self._x_abs_sum = abs_sum
            else:
                self._x_abs_sum += abs_sum
            if self._x_abs_max is None:
                self._x_abs_max = abs_max
            else:
                self._x_abs_max = torch.maximum(self._x_abs_max, abs_max)
            self.nsamples += batch
            if self._omni_x_chunks is not None:
                # 低秩 X 形式：channel-major [in, batch] fp16 CPU，量化期 cat 成 [in, n]
                self._omni_x_chunks.append(inp.detach().to(dtype=torch.float16, device="cpu"))
            self.H = None  # Hessian intentionally unused in rtn mode
            return
        super().add_batch(inp, out)

    def preproc(self) -> None:
        if self.rtn:
            self.preproc_done = True
            return
        super().preproc()

    def _rtn_quantize(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """E8 expert RTN path: plain round-to-nearest, no Hessian compensation.

        Uses the same static per-group scale computation and the same packing
        conventions as the GPTQ path, so packed output tensors are
        format-identical to GPTQ expert quantization (only values differ).
        """
        _ = kwargs
        w = self.layer.weight.data.float().clone().to(self.dev)
        if self.ssz:
            logger.info(
                f"[GPTQ] ssz scale refinement enabled (step={self.ssz_step}) "
                f"for expert {getattr(self.layer, '_expert_idx', '?')}"
            )

        if self.groupsize != -1:
            if not self.static_groups:
                logger.warning("[GPTQ] rtn mode ignores static_groups=False; using static groups")
            if self.ssz:
                logger.warning(
                    "[GPTQ] ssz scale refinement applies to per-channel "
                    "(groupsize=-1) only; skipping for grouped path"
                )
            if self.omni_steps > 0:
                logger.warning(
                    "[GPTQ] omni refine applies to per-channel "
                    "(groupsize=-1) only; skipping for grouped path"
                )
            for i in range(0, self.columns, self.groupsize):
                scale, zero = compute_scales_with_zero(
                    w[:, i : i + self.groupsize], bits=self.wbits, sym=self.sym
                )
                self.scales.append(scale)
                self.zeros.append(zero)
        else:
            s, z = compute_scales_with_zero(
                w, bits=self.wbits, sym=self.sym
            )
            if self.ssz:
                from ..ssz import ssz_calculate_scale

                s = ssz_calculate_scale(w, s, bits=self.wbits, step=self.ssz_step)
            if self.clip_cfg:
                from ..ssz import clip_search_scale

                s = clip_search_scale(
                    w,
                    s,
                    bits=self.wbits,
                    lo=float(self.clip_cfg.get("lo", 0.65)),
                    hi=float(self.clip_cfg.get("hi", 1.0)),
                    steps=int(self.clip_cfg.get("steps", 20)),
                    with_ssz=bool(self.clip_cfg.get("with_ssz", False)),
                    ssz_step=self.ssz_step,
                )
            if self.omni_steps > 0:
                if not self._omni_x_chunks:
                    logger.warning(
                        "[GPTQ] omni refine requested but no expert activations captured; "
                        "keeping clip+SSZ scale"
                    )
                else:
                    from ..ssz import omni_refine_scale

                    x_cm = torch.cat(self._omni_x_chunks, dim=1).to(self.dev)
                    s = omni_refine_scale(
                        w,
                        s,
                        x_cm,
                        bits=self.wbits,
                        steps=self.omni_steps,
                        lr=self.omni_lr,
                    )
            self.scale, self.zero = s, z

        self._omni_x_chunks = None  # freed after per-channel omni consumption
        q = torch.zeros_like(w)
        if self.groupsize != -1:
            for i1 in range(0, self.columns, self.groupsize):
                i2 = min(i1 + self.groupsize, self.columns)
                g = i1 // self.groupsize
                q[:, i1:i2] = quantize_with_scale_zero(
                    w[:, i1:i2], self.scales[g], self.zeros[g], bits=self.wbits
                )
            g_idx = (torch.arange(self.columns, device=self.dev) // self.groupsize).to(
                torch.int32
            )
            final_scale = torch.cat(self.scales, dim=1)
            final_zero = torch.cat(self.zeros, dim=1)
        else:
            q = quantize_with_scale_zero(w, self.scale, self.zero, bits=self.wbits)
            g_idx = torch.zeros(self.columns, dtype=torch.int32, device=self.dev)
            final_scale = self.scale
            final_zero = self.zero

        w_orig = self.layer.weight.data.float().to(self.dev)
        self.layer.weight.data = q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        avg_loss = float(torch.mean((q - w_orig) ** 2).item())
        norm_loss = float(torch.norm(q - w_orig).item())
        self.last_metrics = {
            "rows": self.rows,
            "columns": self.columns,
            "avg_loss": avg_loss,
            "norm_loss": norm_loss,
            "nsamples": float(self.nsamples),
            "fallback": None,
        }

        self.postproc()
        self.free()
        return final_scale.cpu(), final_zero.cpu(), g_idx.cpu()

    def fasterquant(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rtn:
            return self._rtn_quantize(**kwargs)
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

        # g_idx must live on the compute device (self.dev): expert slice weights are
        # offloaded to CPU after statistics collection (BaseHessianAlgorithm
        # _collect_statistics), so self.layer.weight.device may be CPU while
        # perm/invperm/q stay on self.dev. Mixing them breaks g_idx[invperm] on GPU.
        if self.groupsize == -1:
            g_idx = torch.zeros(self.columns, dtype=torch.int32, device=self.dev)
        else:
            if self.static_groups:
                g_idx = (perm // self.groupsize).to(torch.int32)
            else:
                g_idx = (torch.arange(self.columns, device=self.dev) // self.groupsize).to(
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
        act_bits: int = 16,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2/3/4/8 bits are supported.")

        self.infeatures = int(infeatures)
        self.outfeatures = int(outfeatures)
        self.bits = int(bits)
        self.act_bits = int(act_bits)
        self.group_size = self.infeatures if int(group_size) == -1 else int(group_size)
        self.scale_bias = None
        self.maxq = 2**self.bits - 1
        self._is_ascend_format = backend == "npu"

        if self._is_ascend_format:
            if self.bits not in [4, 8]:
                raise ValueError("Ascend backend only supports 4/8-bit GPTQ packing.")
            if self.bits == 8:
                num_groups = math.ceil(self.infeatures / self.group_size)
                self.register_buffer(
                    "weight",
                    torch.zeros((self.outfeatures, self.infeatures), dtype=torch.int8),
                )
                self.register_buffer(
                    "weight_scale",
                    torch.zeros((self.outfeatures, num_groups), dtype=torch.float32),
                )
                self.register_buffer(
                    "weight_offset",
                    torch.zeros((self.outfeatures, num_groups), dtype=torch.float32),
                )
            else:
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
        module_name: Optional[str] = None,
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

        if self.bits == 8:
            self.weight = quant.contiguous().cpu()
            self.weight_scale = scales.to(torch.float32).cpu()
            self.weight_offset = torch.zeros_like(scales, dtype=torch.float32).cpu()
            self.scale_bias = None
            if linear.bias is not None and self.bias is not None:
                self.bias = linear.bias.clone().cpu().to(dtype=self.bias.dtype)
            return

        if self.bits == 4 and self.act_bits == 8:
            # W4A8 row-wise interleave packing: pairs of output rows share one int8.
            # even output row -> low nibble, odd output row -> high nibble (co-worker
            # artifact: int8 [out//2, in] + weight_scale float32 [out, 1]).
            if outfeatures % 2 != 0:
                raise ValueError(
                    f"W4A8 row-wise packing requires even outfeatures, got {outfeatures}"
                )
            # quant 已是二进制补码 [-8,7]。W4A8 kernel 期望补码 nibble（nib = q & 0xF），
            # 不能像 W4A16 那样 + signed_offset 偏置编码（写(+8)/读(-8) 在 W4A16 是对称的，
            # 但 W4A8 kernel 不解 offset，直接按补码解码）。见 W4A8_修改总结.md §10.9。
            q_unsigned = quant.to(torch.uint8) & 0x0F
            packed_u8 = torch.zeros((outfeatures // 2, infeatures), dtype=torch.uint8, device=device)
            packed_u8 |= q_unsigned[0::2]
            packed_u8 |= (q_unsigned[1::2] << 4).to(torch.uint8)
            self.weight = packed_u8.to(torch.int8).contiguous().cpu()
            self.weight_scale = scales.to(torch.float32).cpu()
            self.weight_offset = torch.zeros_like(scales, dtype=torch.float32).cpu()
            # AscendV1 W4A8_DYNAMIC：kernel 无条件把 weight_assist_matrix(scale_bias)
            # Add 进 mm 累加器，补偿补码 nibble 的 -8 偏移，故 sb = 8 * Σ(q_tc * s)。
            # gate/up（列并行）-> [out,1]；down（行并行）-> [out,16]（16 为协议固定
            # 常数、TP 无关，见 §10.9b）。投影类型按 tensor 名判定，与 modelslim
            # process_scale 一致（见 §10.9c）。
            q_scaled = quant.to(torch.float32) * current_scales  # [out, in]
            is_row_parallel = module_name is not None and any(
                tok in module_name for tok in ("down_proj", "o_proj")
            )
            if is_row_parallel:
                if infeatures % 16 != 0:
                    raise ValueError(
                        f"W4A8 down_proj scale_bias requires infeatures divisible by 16, "
                        f"got {infeatures}"
                    )
                chunk = infeatures // 16
                scale_bias = 8.0 * q_scaled.reshape(outfeatures, 16, chunk).sum(dim=-1)
            else:
                scale_bias = 8.0 * q_scaled.sum(dim=1, keepdim=True)
            self.scale_bias = scale_bias.contiguous().cpu()
            if linear.bias is not None and self.bias is not None:
                self.bias = linear.bias.clone().cpu().to(dtype=self.bias.dtype)
            return

        quant_unsigned = (quant + signed_offset).to(torch.uint8)

        packed = torch.zeros((outfeatures, infeatures // 8), dtype=torch.int32, device=device)
        for i in range(8):
            cols = torch.arange(i, infeatures, 8, device=device)
            packed |= quant_unsigned[:, cols].to(torch.int32) << (self.bits * i)

        self.weight = packed.contiguous().cpu()
        self.weight_scale = scales.to(torch.bfloat16).cpu()
        self.weight_offset = torch.zeros_like(zeros, dtype=torch.bfloat16).cpu()
        self.scale_bias = None
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
        module_name: Optional[str] = None,
    ) -> None:
        if self._is_ascend_format:
            self._pack_ascend(linear, scales, zeros, g_idx, module_name)
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
        act_bits: int = 16,
        layer_routing: Optional[List[Dict]] = None,
        max_calib_samples: int = 128,
        save_backend: Optional[str] = None,
        expert_chunk_size: int = 1,
        quantize_mtp: bool = False,
        save_mtp_debug: bool = False,
        scale_search: Optional[Dict] = None,
        smooth_quant: Optional[Dict] = None,
        expert_mode: Optional[str] = None,
        ssz_step: int = 10,
        expert_clip_search: Optional[Dict] = None,
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
        self.act_bits = int(act_bits)
        self.layer_routing = list(layer_routing) if layer_routing else None
        self._save_backend = save_backend
        self.scale_search_cfg = scale_search
        # SmoothQuant (norm-linear) preprocessing: fold s = |X|^α / |W|^(1-α)
        # into norm gain (÷s) + consumer linear weights (×s on input columns).
        # Matches msmodelslim flex_smooth_quant NormLinearSubgraph semantics
        # (per-input-channel scale, single s per norm across all targets).
        self.smooth_quant_cfg = dict(smooth_quant) if smooth_quant else None
        self._smooth_norm_stats: Dict[str, list] = {}
        self._smooth_norm_act: Dict[str, list] = {}
        # Layer-1.5: producer->consumer subgraph SQ (up_down / linear_linear / ov).
        # key=f"{layer_name}.{consumer_path}" -> (stype, sub_cfg, acc_list)
        self._smooth_subgraph_stats: Dict[str, tuple] = {}
        # FlexSmooth alpha/beta search inputs: per-subgraph activation samples
        # (bounded token budget), used by alpha_beta_search to pick (α, β).
        self._smooth_subgraph_act: Dict[str, list] = {}
        # E8: expert-module quantization mode. None -> per-expert GPTQ (default);
        # "rtn" -> skip Hessian error compensation for MoE expert modules, plain
        # RTN quantization on the same scale/pack path (modelslim "AWQ + no GPTQ"
        # structural counterpart). Dense modules are unaffected.
        # "ssz" -> rtn base + SSZ per-channel scale refinement (iterative LSQ,
        # ref: msmodelslim core/quantizer/impl/ssz.py; modelslim W4A8 experts).
        if expert_mode is None:
            self.expert_mode = None
        else:
            expert_mode = str(expert_mode).lower()
            if expert_mode not in ("rtn", "ssz"):
                raise ValueError(
                    f"[GPTQ] unsupported expert_mode: {expert_mode!r} "
                    f"(expected None, 'rtn' or 'ssz')"
                )
            self.expert_mode = expert_mode
        # Both rtn and ssz skip expert Hessian; ssz additionally refines the
        # per-channel scale (SSZ) before quantization.
        self._expert_rtn = self.expert_mode in ("rtn", "ssz")
        self._expert_ssz = self.expert_mode == "ssz"
        self.ssz_step = int(ssz_step)
        # Opt-in per-channel clip-search on the expert (W4) scale (default off).
        # Algorithm + rationale: ssz.clip_search_scale.
        self._clip_cfg = dict(expert_clip_search) if expert_clip_search else None

    @property
    def _ascend_quant_type(self) -> str:
        if self.fake_quant:
            return "FLOAT"
        if self.act_bits == 8:
            return f"W{self.wbits}A8_DYNAMIC"
        return f"W{self.wbits}A16"

    def _resolve_layer_wbits(self, module_rel_name: str) -> int:
        """Resolve per-layer weight bits via layer_routing glob patterns.

        Callers pass two name forms:
          - full tensor/module paths:  "model.layers.9.self_attn.q_a_proj"
          - layer-relative names:      "self_attn.q_a_proj"
        Patterns are matched against both forms so that handler creation
        (relative names) and metadata annotation (full names) stay consistent.
        """
        if not self.layer_routing:
            return self.wbits
        candidates = (module_rel_name, f"*.{module_rel_name}")
        for entry in self.layer_routing:
            patterns = entry.get("patterns") or []
            if any(
                fnmatch.fnmatch(candidate, pattern)
                for candidate in candidates
                for pattern in patterns
            ):
                return int(entry.get("wbits", self.wbits))
        return self.wbits

    @staticmethod
    def _strip_tensor_suffix(name: str) -> str:
        for suffix in (".weight_scale", ".weight_offset", ".scale_bias", ".weight", ".bias"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _tensor_quant_type(self, name: str) -> str:
        if self.target_backend != "npu":
            return self._quantized_type_label
        if self.fake_quant:
            return "FLOAT"
        wbits = self._resolve_layer_wbits(self._strip_tensor_suffix(name))
        if self.act_bits == 8:
            return f"W{wbits}A8_DYNAMIC"
        return f"W{wbits}A16"

    def _finalize_chunk_metadata(self, chunk, quantized_tensor_names: set[str]) -> None:
        """Per-tensor type annotation for mixed (W4A8 + W8A8) outputs."""
        tensor_types = {name: "FLOAT" for name in chunk.all_tensors().keys()}
        for name in quantized_tensor_names:
            if name in tensor_types:
                tensor_types[name] = self._tensor_quant_type(name)
        chunk.metadata["tensor_types"] = tensor_types

    def _log_start_params(self) -> None:
        logger.info(
            f"[{self._TAG}] start: "
            f"wbits={self.wbits}, groupsize={self.groupsize}, "
            f"actorder={self.actorder}, percdamp={self.percdamp}, "
            f"act_bits={self.act_bits}"
        )
        if self.scale_search_cfg:
            logger.info(
                f"[{self._TAG}] scale_search enabled: {self.scale_search_cfg}"
            )
        if self.smooth_quant_cfg:
            logger.info(f"[{self._TAG}] smooth_quant enabled: {self.smooth_quant_cfg}")
        if self.expert_mode:
            logger.info(
                f"[{self._TAG}] expert_mode={self.expert_mode}: MoE expert modules "
                f"quantized with {self.expert_mode.upper()} (no Hessian compensation); "
                f"dense modules unchanged"
            )
            if self._expert_ssz:
                logger.info(
                    f"[{self._TAG}] ssz scale refinement step={self.ssz_step} "
                    f"(iterative LSQ per-channel scale, ref: msmodelslim ssz.py)"
                )

    def _apply_scale_search(self, layer, handlers, chunk) -> None:
        """Apply AWQ scale search to MoE expert handlers before GPTQ quantization.

        For each expert handler, computes per-input-channel activation abs-mean
        from calibration statistics, grid-searches the optimal AWQ scale s, and
        replaces the 3D parameter with the adjusted weight: w_adjusted = RTN(w*s)/s.

        This is a float weight with improved quantization properties; GPTQ
        fasterquant then quantizes it normally. No post-quantize "divide by s"
        step is needed because s is folded into w_adjusted's values.

        SmoothQuant (smooth_quant) preprocessing runs AFTER AWQ, matching
        modelslim's processor order (flex_awq_ssz -> flex_smooth_quant).
        """
        if not self.scale_search_cfg and not self.smooth_quant_cfg:
            return

        moe_cfg = (self.scale_search_cfg or {}).get("moe")
        method = (moe_cfg or {}).get("method", "awq")
        if moe_cfg and method == "updown_awq":
            self._apply_updown_awq(layer, handlers, moe_cfg)
        elif moe_cfg and method == "awq":
            nbits = int(moe_cfg.get("nbits", self.wbits))
            grid = int(moe_cfg.get("grid", 20))
            search_stat = str(moe_cfg.get("search_stat", "mean")).lower()

            from ..scale_search import awq_search, awq_apply

            import time as _time
            t_search = 0.0
            n_search = 0

            for handler in handlers.values():
                is_batched = getattr(handler, "_is_batched_expert", False)
                is_expert = (
                    is_batched
                    or getattr(getattr(handler, "layer", None), "_is_expert_slice", False)
                )
                if not is_expert:
                    continue

                x_abs_mean = getattr(handler, "x_abs_mean", None)
                if x_abs_mean is None:
                    continue

                if is_batched:
                    xax = (
                        getattr(handler, "x_abs_max", None)
                        if search_stat == "max"
                        else None
                    )
                    for local_e in range(handler._CH):
                        sl = handler._slice_linears[local_e]
                        w3d = sl._weight_3d
                        e = sl._expert_idx
                        w = w3d.data[e].float()
                        xm = x_abs_mean[local_e].to(w.device)
                        xm_search = (
                            xax[local_e].to(w.device) if xax is not None else None
                        )
                        _t0 = _time.perf_counter()
                        s = awq_search(w, xm, nbits=nbits, grid=grid, x_search=xm_search)
                        w_adjusted = awq_apply(w, s, nbits=nbits)
                        t_search += _time.perf_counter() - _t0
                        n_search += 1
                        w3d.data[e].copy_(w_adjusted.to(w.dtype))
                        # Update the slice linear's weight view to the new data
                        sl.weight = torch.nn.Parameter(
                            w3d.data[e], requires_grad=False
                        )
                else:
                    sl = handler.layer
                    w3d = sl._weight_3d
                    e = sl._expert_idx
                    w = w3d.data[e].float()
                    xm_search = None
                    if search_stat == "max":
                        xax = getattr(handler, "x_abs_max", None)
                        xm_search = xax.to(w.device) if xax is not None else None
                    _t0 = _time.perf_counter()
                    s = awq_search(
                        w,
                        x_abs_mean.to(w.device),
                        nbits=nbits,
                        grid=grid,
                        x_search=xm_search,
                    )
                    w_adjusted = awq_apply(w, s, nbits=nbits)
                    t_search += _time.perf_counter() - _t0
                    n_search += 1
                    w3d.data[e].copy_(w_adjusted.to(w.dtype))
                    sl.weight = torch.nn.Parameter(
                        w3d.data[e], requires_grad=False
                    )

            if n_search:
                logger.info(
                    f"[{self._TAG}] AWQ scale_search: {layer.name}: {n_search} experts, "
                    f"{t_search:.2f}s total, {t_search / n_search * 1000:.1f}ms/expert"
                )

        if self.smooth_quant_cfg:
            self._apply_smooth_quant(layer, chunk)

    def _apply_updown_awq(self, layer, handlers, moe_cfg) -> None:
        """up-down middle-channel AWQ (modelslim flex_awq_ssz UpDown subgraph).

        Scale s is searched per middle channel from the DOWN projection's input
        activation abs-mean (= swiglu output, i.e. the middle activation — the
        down handler's x_abs_mean already IS this statistic). Then folded as a
        balanced scale (kept scaled, matching modelslim UpDownSubgraphFusion):
          - down:     W_dn <- W_dn * s_col          (input-column scale, kept)
          - gate_up:  up half rows [I, 2I) <- W_gu[I:2I] * (1/s)
        s cancels in the overall computation (down x s · up x 1/s) and the normal
        W4 quantization acts on the scaled W_dn (the AWQ benefit). The gate half
        (rows [0, I)) is left untouched (swiglu nonlinearity).

        Ref: msmodelslim processor/anti_outlier/flex_smooth/api.py::
             flex_awq_ssz_impl_up_down + common/subgraph_fusion.py::UpDownSubgraphFusion.
        """
        import time as _time

        from ..scale_search import awq_apply_scaled, awq_search

        nbits = int(moe_cfg.get("nbits", self.wbits))
        grid = int(moe_cfg.get("grid", 20))
        # AWQ alpha-search statistic: modelslim flex_awq_ssz searches alpha with
        # max|act|^alpha and applies the final scale with mean|act|^alpha
        # (processor/anti_outlier/flex_smooth/api.py::flex_awq_ssz_impl_up_down
        # -> search_scale/transform). "mean" keeps the legacy mean/mean behavior.
        search_stat = str(moe_cfg.get("search_stat", "mean")).lower()
        if search_stat not in ("mean", "max"):
            raise ValueError(
                f"scale_search.moe.search_stat must be 'mean' or 'max', got {search_stat!r}"
            )

        # Per-expert context: (experts_path, expert_idx) -> down w3d + middle mean
        experts_ctx: Dict[tuple, Dict] = {}
        n_skip = 0
        for handler in handlers.values():
            is_batched = getattr(handler, "_is_batched_expert", False)
            if not (
                is_batched
                or getattr(getattr(handler, "layer", None), "_is_expert_slice", False)
            ):
                continue
            # _proj_type: batched 在 handler 上；per-expert 在 handler.layer（slice linear）上
            proj_type = (
                handler._proj_type if is_batched
                else getattr(handler.layer, "_proj_type", None)
            )
            if proj_type != "down_proj":
                continue
            # 未在校准中被路由到的专家 x_abs_mean 为 None（scale search 统计缺失）
            # → 跳过（该专家不做 AWQ，保持原权重），与纯 "awq" 分支的 None 处理一致。
            # 缺此保护会崩掉整个量化（GLM layer 57 实测，2026-09-12）。
            if is_batched:
                xam = handler.x_abs_mean
                if xam is None:
                    n_skip += 1
                    continue
                xax = getattr(handler, "x_abs_max", None)
                for local_idx in range(handler._CH):
                    e = handler._expert_start_idx + local_idx
                    ctx = experts_ctx.setdefault(
                        (handler._experts_module_path, e), {}
                    )
                    ctx["down_w3d"] = handler._slice_linears[0]._weight_3d
                    ctx["mid_mean"] = xam[local_idx].cpu() / 1.0
                    if xax is not None:
                        ctx["mid_max"] = xax[local_idx].cpu()
            else:
                xam = handler.x_abs_mean
                if xam is None:
                    n_skip += 1
                    continue
                sl = handler.layer
                ctx = experts_ctx.setdefault(
                    (sl._experts_module_path, sl._expert_idx), {}
                )
                ctx["down_w3d"] = sl._weight_3d
                ctx["mid_mean"] = xam.cpu() / 1.0
                xax = getattr(handler, "x_abs_max", None)
                if xax is not None:
                    ctx["mid_max"] = xax.cpu()

        if not experts_ctx:
            return

        # Resolve gate_up 3D per experts_module path (fused gate_up_proj)
        gate_up_by_path: Dict[str, nn.Parameter] = {}
        for handler in handlers.values():
            is_batched = getattr(handler, "_is_batched_expert", False)
            if not (
                is_batched
                or getattr(getattr(handler, "layer", None), "_is_expert_slice", False)
            ):
                continue
            proj_type = (
                handler._proj_type if is_batched
                else getattr(handler.layer, "_proj_type", None)
            )
            if proj_type != "gate_up_proj":
                continue
            if getattr(handler, "_is_batched_expert", False):
                gate_up_by_path[handler._experts_module_path] = (
                    handler._slice_linears[0]._weight_3d
                )
            else:
                sl = handler.layer
                gate_up_by_path[sl._experts_module_path] = sl._weight_3d

        t_search = 0.0
        n_search = 0
        for (experts_path, e), ctx in experts_ctx.items():
            down_w3d = ctx["down_w3d"]
            mid = ctx["mid_mean"]
            mid_search = ctx.get("mid_max") if search_stat == "max" else None
            w = down_w3d.data[e].float()
            _t0 = _time.perf_counter()
            s = awq_search(w, mid, nbits=nbits, grid=grid, x_search=mid_search)
            w_adjusted = awq_apply_scaled(w, s)  # w*s, columns (scale-only, kept scaled)
            t_search += _time.perf_counter() - _t0
            n_search += 1
            down_w3d.data[e].copy_(w_adjusted.to(down_w3d.dtype))

            gu_w3d = gate_up_by_path.get(experts_path)
            if gu_w3d is not None:
                I = gu_w3d.shape[1] // 2
                inv_s = (1.0 / s).to(gu_w3d.dtype).view(-1, 1)
                gu = gu_w3d.data[e].float()
                gu[I:] = gu[I:] * inv_s
                gu_w3d.data[e].copy_(gu.to(gu_w3d.dtype))
            else:
                logger.warning(
                    f"[{self._TAG}] updown_awq: no fused gate_up_proj for "
                    f"experts_module {experts_path}, expert {e} (up-half fold skipped)"
                )

        if n_search or n_skip:
            _skip = f" | skipped {n_skip} un-routed (no x_abs_mean)" if n_skip else ""
            _per = (t_search / n_search * 1000) if n_search else 0.0
            logger.info(
                f"[{self._TAG}] updown_awq: {layer.name}: {n_search} experts, "
                f"{t_search:.2f}s total, {_per:.1f}ms/expert{_skip}"
            )

    # ---------------- SmoothQuant (norm-linear) ----------------
    #
    # Ref: msmodelslim processor/anti_outlier/flex_smooth/api.py::
    #      flex_smooth_impl_norm_linear + common/subgraph_fusion.py::
    #      NormLinearSubgraphFusion, and model/glm_5/quarot.py::get_ln_fuse_map.
    #
    # Scale s = |X|^α / |W|^(1-α) per input channel (α default 0.5, no search):
    #   - w_scale: per-input-channel abs-max across ALL target linears of the
    #     same norm (stacked, then max) — modelslim compute_multi_weight_scale.
    #   - x_scale: per-channel abs-max of the norm OUTPUT (collected via hook).
    #   - fold: norm gain γ ← γ/s ; every target linear weight cols ← ×s.
    # The net function is identical; quantization happens on the smoothed W*s.
    #
    # Default scope (GLM-5.2, matching best_practice flex_smooth_quant
    # "self_attn" + "input_layernorm"; post_attention_layernorm excluded):
    #   input_layernorm -> q_a_proj, kv_a_proj_with_mqa
    #   self_attn.q_a_layernorm -> self_attn.q_b_proj
    #   self_attn.kv_a_layernorm -> self_attn.kv_b_proj

    @staticmethod
    def _try_get_child(layer, path):
        """_get_child_module that tolerates missing modules (returns None)."""
        try:
            return _get_child_module(layer, path)
        except AttributeError:
            return None

    DEFAULT_SMOOTH_NORM_TARGETS = [
        {"norm": "input_layernorm",
         "targets": ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]},
        {"norm": "self_attn.q_a_layernorm", "targets": ["self_attn.q_b_proj"]},
        {"norm": "self_attn.kv_a_layernorm", "targets": ["self_attn.kv_b_proj"]},
    ]

    def _smooth_norm_targets(self) -> list:
        if not self.smooth_quant_cfg:
            return []
        # Explicit "targets" key (even empty) overrides the default; key-absent
        # falls back to DEFAULT so existing recipes keep their norm-linear SQ.
        if "targets" in self.smooth_quant_cfg:
            return self.smooth_quant_cfg["targets"]
        return list(self.DEFAULT_SMOOTH_NORM_TARGETS)

    def _smooth_subgraphs(self) -> list:
        """Layer-1.5 producer->consumer SQ subgraphs (opt-in via smooth_quant.subgraphs).

        Each entry: {type: up_down|linear_linear|ov, <producer key>, <consumer key>}.
        Keys:
          up_down:       producer=up, consumer=down   (gate untouched; routed experts use updown_awq)
          linear_linear: producer=l1, consumer=l2      (y=l2(l1(x)))
          ov:            producer=v, consumer=o        (MLA value->output)
        """
        if not self.smooth_quant_cfg:
            return []
        return self.smooth_quant_cfg.get("subgraphs", []) or []

    def _collect_statistics(self, layer_module, handlers, *, layer_name, chunk_index):
        """收集 SmoothQuant norm 输出 per-channel abs-max（叠加到父类统计收集）。"""
        if not self.smooth_quant_cfg:
            return super()._collect_statistics(
                layer_module, handlers, layer_name=layer_name, chunk_index=chunk_index
            )
        self._smooth_norm_stats.clear()
        self._smooth_norm_act.clear()
        hooks = []
        for entry in self._smooth_norm_targets():
            norm_module = self._try_get_child(layer_module, entry["norm"])
            if norm_module is None or not hasattr(norm_module, "weight"):
                continue
            acc = [None]  # mutable holder: [hidden] running max, lazily created
            act_holder: list = []
            self._smooth_norm_stats[f"{layer_name}.{entry['norm']}"] = acc
            self._smooth_norm_act[f"{layer_name}.{entry['norm']}"] = act_holder

            def _norm_hook(module, inp, out, _acc=acc, _hold=act_holder):
                self._smooth_collect(out, _acc)
                self._smooth_collect_act(out, _hold)
                return None  # ★ 必须返回 None：forward_hook 返回非 None 会替换模块输出

            hooks.append(norm_module.register_forward_hook(_norm_hook))
        # Layer-1.5 subgraph intermediate capture: pre-hook on each consumer module,
        # recording per-channel abs-max of its INPUT (= producer output / the
        # intermediate activation being smoothed).
        self._smooth_subgraph_stats.clear()
        self._smooth_subgraph_act.clear()
        for sub in self._smooth_subgraphs():
            stype = sub.get("type")
            if stype == "ov":
                consumer_path = sub.get("o") or sub.get("consumer")
            elif stype == "up_down":
                consumer_path = sub.get("down") or sub.get("consumer")
            elif stype == "linear_linear":
                consumer_path = sub.get("l2") or sub.get("consumer")
            else:
                continue
            if not consumer_path:
                continue
            consumer_mod = self._try_get_child(layer_module, consumer_path)
            if consumer_mod is None or not hasattr(consumer_mod, "weight"):
                continue
            acc = [None]
            key = f"{layer_name}.{consumer_path}"
            act_holder: list = []
            self._smooth_subgraph_stats[key] = (stype, sub, acc)
            self._smooth_subgraph_act[key] = act_holder

            def _pre_hook(module, args, _acc=acc, _hold=act_holder):
                if args and isinstance(args[0], torch.Tensor):
                    self._smooth_collect(args[0], _acc)
                    self._smooth_collect_act(args[0], _hold)

            hooks.append(consumer_mod.register_forward_pre_hook(_pre_hook))
        try:
            return super()._collect_statistics(
                layer_module, handlers, layer_name=layer_name, chunk_index=chunk_index
            )
        finally:
            for h in hooks:
                h.remove()

    @staticmethod
    def _smooth_collect(out: torch.Tensor, acc: list) -> None:
        """Running per-channel abs-max of a norm output into a [hidden] holder."""
        t = out.float().abs()
        while t.dim() > 1:
            t = t.amax(dim=0)
        if acc[0] is None:
            acc[0] = torch.zeros_like(t)
        acc[0].copy_(torch.maximum(acc[0].to(t.device), t).to(acc[0].device))

    def _smooth_collect_act(self, out: torch.Tensor, holder: list) -> None:
        """Collect up to ``act_sample_tokens`` activation rows for the α/β search.

        Stored on CPU in bf16 (the searcher casts to float32). Total per subgraph
        is bounded by the config's ``act_sample_tokens`` (default 512) so the
        grid search stays cheap; the reference uses the full activations.
        """
        cap = int((self.smooth_quant_cfg or {}).get("act_sample_tokens", 512))
        have = sum(int(t.shape[0]) for t in holder)
        if have >= cap:
            return
        x = out.detach()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        n = min(int(x.shape[0]), cap - have)
        if n <= 0:
            return
        holder.append(x[:n].to("cpu", dtype=torch.bfloat16))

    def _check_scale_health(self, s: torch.Tensor, where: str) -> None:
        """Warn when a SmoothQuant scale leaves the healthy range [1e-2, 1e2].

        An extremely small s scales the consumer's weight columns to ~0 so the
        int8 weight quant rounds them away (dead channels → garbage output); it
        almost always indicates a broken activation statistic (e.g. the missing
        ``abs`` in ``_smooth_collect``, fixed 2026-09-12).
        """
        n_small = int((s < 1e-2).sum())
        n_large = int((s > 1e2).sum())
        if n_small or n_large:
            logger.warning(
                f"[{self._TAG}] smooth_quant scale health {where}: "
                f"{n_small} ch < 1e-2, {n_large} ch > 1e2 "
                f"(min={float(s.min()):.3e}, max={float(s.max()):.3e}, n={s.numel()})"
            )

    def _flex_smooth_scale(self, act_key, x_scale, w_scale, search_weights,
                           default_alpha: float, where: str):
        """Layer-1.5 SQ scale: searched (alpha,beta) when enabled, else fixed alpha.

        msmodelslim's flex_smooth_quant leaves alpha/beta unset in the production
        recipe, so each subgraph runs ``FlexSmoothAlphaBetaSearcher`` (grid step
        0.05: phase-1 searches alpha with beta=1-alpha, phase-2 searches beta).
        Opt-in via ``smooth_quant.alpha_beta_search: true``.
        """
        from ..scale_search import smooth_scale, smooth_scale_ab

        cfg = self.smooth_quant_cfg or {}
        if cfg.get("alpha_beta_search"):
            act = (self._smooth_subgraph_act.get(act_key)
                   or self._smooth_norm_act.get(act_key))
            if act:
                from ..alpha_beta_search import FlexSmoothAlphaBetaSearcher

                act_t = torch.cat(act, dim=0).float().cpu()
                w_t = search_weights.detach().float().cpu()
                searcher = FlexSmoothAlphaBetaSearcher(
                    act_sym=True, search_step=float(cfg.get("search_step", 0.05))
                )
                a, b, mse = searcher.search_alpha_beta(act_t, w_t)
                logger.info(
                    f"[{self._TAG}] flex_smooth a/b search {where}: "
                    f"alpha={a:.2f} beta={b:.2f} mse={mse:.6f} "
                    f"(tokens={int(act_t.shape[0])})"
                )
                return smooth_scale_ab(x_scale, w_scale, a, b)
            logger.warning(
                f"[{self._TAG}] alpha_beta_search enabled but no act samples for "
                f"{where}; fallback fixed alpha={default_alpha}"
            )
        return smooth_scale(x_scale, w_scale, default_alpha)

    def _apply_smooth_quant(self, layer, chunk) -> None:
        """SmoothQuant fold for norm-linear pairs (post AWQ, pre quantization).

        Folds s into 2D linear weights (columns ×s) and the norm gain (÷s),
        then syncs the (FLOAT pass-through) norm snapshot so the saver persists
        the smoothed γ — matching w4a8c8's non-all-1 norm values (§1.6.2 #3).
        """
        from ..scale_search import compute_weight_scale, smooth_scale

        alpha = float(self.smooth_quant_cfg.get("alpha", 0.5))
        # 真实管线 layer 是 LayerInfo（chunk 数据类），模块需从运行时模型解析；
        # 单测可能直接传 nn.Module，兼容两种。
        if isinstance(layer, nn.Module):
            layer_module = layer
        else:
            layer_module = _get_child_module(self._runtime_model, layer.name)
        n_pairs = 0
        for entry in self._smooth_norm_targets():
            norm_path = entry["norm"]
            norm_module = self._try_get_child(layer_module, norm_path)
            if norm_module is None or not hasattr(norm_module, "weight"):
                continue
            acc = self._smooth_norm_stats.get(f"{layer.name}.{norm_path}")
            if acc is None or acc[0] is None:
                continue
            x_max = acc[0]

            target_weights = []
            target_modules = []
            for tname in entry["targets"]:
                tmod = self._try_get_child(layer_module, tname)
                if tmod is None or not hasattr(tmod, "weight"):
                    continue
                target_modules.append(tmod)
                target_weights.append(tmod.weight.detach().float())
            if not target_weights:
                continue

            w_max = torch.stack(
                [compute_weight_scale(w) for w in target_weights]
            ).amax(dim=0)  # per-input-channel max across all targets
            s = self._flex_smooth_scale(
                f"{layer.name}.{norm_path}", x_max, w_max,
                torch.cat(target_weights, dim=0), alpha,
                f"{layer.name}:{norm_path}",
            )
            self._check_scale_health(s, f"{layer.name}:{norm_path}")

            for tmod in target_modules:
                w = tmod.weight.detach().float()
                tmod.weight.data.copy_(
                    (w * s.to(w.device).view(1, -1)).to(tmod.weight.dtype)
                )
            gamma = norm_module.weight.detach().float()
            norm_module.weight.data.copy_(
                (gamma / s.to(gamma.device)).to(norm_module.weight.dtype)
            )

            # Sync FLOAT pass-through norm snapshot (chunk copy != in-memory module)
            if chunk is not None:
                snap_name = f"{layer.name}.{norm_path}.weight"
                snap = chunk.get_tensor(snap_name)
                if snap is not None:
                    chunk.update_tensor(
                        snap_name, norm_module.weight.detach().cpu()
                    )
            n_pairs += 1

        # --- Layer-1.5 subgraph SQ folds (up_down / linear_linear / ov) ---
        n_sub = self._apply_subgraph_sq(layer, layer_module, alpha, chunk)

        if n_pairs or n_sub:
            logger.info(
                f"[{self._TAG}] smooth_quant: {layer.name}: {n_pairs} norm-linear + "
                f"{n_sub} subgraph pairs folded (alpha={alpha})"
            )

    def _apply_subgraph_sq(self, layer, layer_module, alpha, chunk) -> int:
        """Fold producer->consumer SQ subgraphs (Layer-1.5). Returns #folded.

        Consumes the per-consumer intermediate abs-max captured in
        ``_smooth_subgraph_stats`` (pre-hooks) + the consumer weight scale,
        computes s = smooth_scale(x, w, alpha), then folds:
          consumer.weight input-col  *= s
          producer.weight  output-row *= 1/s
        (ov additionally aligns the MLA v-channel head layout.)
        """
        from ..scale_search import compute_weight_scale, smooth_scale

        n_sub = 0
        for sub in self._smooth_subgraphs():
            stype = sub.get("type")
            if stype == "up_down":
                prod_path = sub.get("up") or sub.get("producer")
                cons_path = sub.get("down") or sub.get("consumer")
            elif stype == "linear_linear":
                prod_path = sub.get("l1") or sub.get("producer")
                cons_path = sub.get("l2") or sub.get("consumer")
            elif stype == "ov":
                prod_path = sub.get("v") or sub.get("producer")
                cons_path = sub.get("o") or sub.get("consumer")
            else:
                continue
            if not prod_path or not cons_path:
                continue
            acc_entry = self._smooth_subgraph_stats.get(f"{layer.name}.{cons_path}")
            if acc_entry is None:
                continue
            acc = acc_entry[2]
            if acc is None or acc[0] is None:
                logger.warning(
                    f"[{self._TAG}] smooth_quant subgraph {stype}: no intermediate "
                    f"stats for {layer.name}.{cons_path} (skip)"
                )
                continue
            x_scale = acc[0].float()
            cons_mod = self._try_get_child(layer_module, cons_path)
            prod_mod = self._try_get_child(layer_module, prod_path)
            if cons_mod is None or prod_mod is None:
                continue
            w_scale = compute_weight_scale(cons_mod.weight.detach())
            if w_scale.numel() != x_scale.numel():
                # ov producer/consumer channel mismatch is handled by the caller via
                # head-dim reshape; for the clean cases dims must match.
                if stype != "ov":
                    logger.warning(
                        f"[{self._TAG}] smooth_quant subgraph {stype}: dim mismatch "
                        f"x={x_scale.numel()} w={w_scale.numel()} (skip)"
                    )
                    continue
            s = self._flex_smooth_scale(
                f"{layer.name}.{cons_path}", x_scale, w_scale,
                cons_mod.weight.detach(), alpha, f"{layer.name}:{cons_path}",
            )
            self._check_scale_health(s, f"{layer.name}:{cons_path}")
            if stype == "ov":
                self._sq_fold_ov(layer_module, cons_mod, prod_mod, s,
                                 sub.get("num_heads"), sub.get("kv_heads"),
                                 prod_path, cons_path)
            else:
                self._sq_fold_pair(cons_mod, prod_mod, s)
            # ★ Sync chunk snapshots for the FLOAT pass-through producer/consumer.
            # e.g. kv_b_proj is FLOAT (model default skip); its in-memory fold must be
            # written back to the chunk or the saved artifact keeps the ORIGINAL weight
            # while o_proj (quantized) is folded → ov function broken → garbage output
            # (found 2026-09-12). Quantized targets are harmlessly overwritten by packing.
            # Mirrors the norm-linear fold's norm snapshot sync.
            if chunk is not None:
                for _p, _m in ((prod_path, prod_mod), (cons_path, cons_mod)):
                    for _suf in ("weight", "bias"):
                        _attr = getattr(_m, _suf, None)
                        if _attr is None:
                            continue
                        _n = f"{layer.name}.{_p}.{_suf}"
                        if chunk.get_tensor(_n) is not None:
                            chunk.update_tensor(_n, _attr.detach().cpu())
            n_sub += 1
        return n_sub

    @staticmethod
    def _sq_fold_pair(cons_mod, prod_mod, s) -> None:
        """SmoothQuant fold for a clean producer->consumer linear pair.

        consumer.weight input-col *= s ; producer.weight output-row *= 1/s
        (+ producer.bias *= 1/s if present). Net function preserved.
        """
        from ..scale_search import awq_apply_scaled
        s_dev = s.to(cons_mod.weight.device)
        cdtype = cons_mod.weight.dtype
        pdtype = prod_mod.weight.dtype
        cons_mod.weight.data.copy_(
            awq_apply_scaled(cons_mod.weight.detach(), s_dev).to(cdtype)
        )
        inv = (1.0 / s_dev).to(pdtype)
        prod_mod.weight.data.copy_(
            (prod_mod.weight.detach().float() * inv.view(-1, 1)).to(pdtype)
        )
        if prod_mod.bias is not None:
            prod_mod.bias.data.copy_(
                (prod_mod.bias.detach().float() * inv).to(prod_mod.bias.dtype)
            )

    def _sq_fold_ov(self, layer_module, o_mod, v_mod, s, num_heads, kv_heads,
                    v_path, o_path) -> None:
        """MLA ov fold: o_proj input-col *= s ; kv_b_proj v-output-row *= 1/s.

        v is the last ``v_head_dim`` channels of each head group in kv_b_proj's
        output [num_heads*(qk_nope+v_head)]. Requires v_mod == kv_b_proj and a
        head-structured s; falls back to the clean pair fold when shapes align.
        """
        from ..scale_search import awq_apply_scaled
        try:
            cfg = getattr(self._runtime_model, "config", None) or getattr(layer_module, "config", None)
            v_head = getattr(cfg, "v_head_dim", None)
            qk_nope = getattr(cfg, "qk_nope_head_dim", None)
            nh = num_heads or getattr(cfg, "num_attention_heads", None)
            if v_head and qk_nope and nh:
                per_head = qk_nope + v_head
                # s is per o_proj input channel = [num_heads * v_head]; reshape to
                # the per-head v slice [nh, v_head] and scale those kv_b_proj rows.
                s_v = s.to(v_mod.weight.device).float().view(nh, v_head)
                # o_proj input-col ×s
                o_mod.weight.data.copy_(
                    awq_apply_scaled(o_mod.weight.detach(), s.to(o_mod.weight.device)).to(o_mod.weight.dtype)
                )
                inv = (1.0 / s_v)
                w = v_mod.weight.detach().float().view(nh, per_head, -1)
                w[:, -v_head:, :] = w[:, -v_head:, :] * inv.unsqueeze(-1)
                v_mod.weight.data.copy_(w.reshape(v_mod.weight.shape).to(v_mod.weight.dtype))
                if v_mod.bias is not None:
                    vb = v_mod.bias.detach().float().view(nh, per_head)
                    vb[:, -v_head:] = vb[:, -v_head:] * inv
                    v_mod.bias.data.copy_(vb.reshape(v_mod.bias.shape).to(v_mod.bias.dtype))
                return
        except Exception as e:  # noqa: BLE001 - never break quant on an ov edge case
            logger.warning(f"[{self._TAG}] smooth_quant ov: head-dim fold failed ({e}); "
                           f"falling back to clean pair fold")
        self._sq_fold_pair(o_mod, v_mod, s)

    def _pack_quant_linear_tensors(
        self,
        module_name: str,
        linear_module: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        wbits: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        bits = self.wbits if wbits is None else int(wbits)
        quant_linear = GPTQQuantLinear(
            bits=bits,
            group_size=self.groupsize,
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            bias=linear_module.bias is not None,
            weight_dtype=linear_module.weight.dtype,
            backend=self.target_backend,
            act_bits=self.act_bits,
        )
        proxy = SimpleNamespace(
            weight=linear_module.weight.detach().cpu(),
            bias=(linear_module.bias.detach().cpu() if linear_module.bias is not None else None),
        )
        quant_linear.pack(proxy, scales.cpu(), zeros.cpu(), g_idx.cpu(), module_name=module_name)

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
            if getattr(quant_linear, "scale_bias", None) is not None:
                tensors[f"{module_name}.scale_bias"] = quant_linear.scale_bias.cpu()
                quantized_names.append(f"{module_name}.scale_bias")
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
                            wbits=self._resolve_layer_wbits(module_rel_name),
                            groupsize=self.groupsize,
                            blocksize=self.blocksize,
                            sym=self.sym,
                            percdamp=self.percdamp,
                            preproc_hessian=self.preproc_hessian,
                            hessian_device=torch.device("cpu"),
                            rtn=self._expert_rtn,
                            ssz=self._expert_ssz,
                            ssz_step=self.ssz_step,
                            clip_search=self._clip_cfg,
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
                            wbits=self._resolve_layer_wbits(expert_rel_name),
                            groupsize=self.groupsize,
                            blocksize=self.blocksize,
                            actorder=self.actorder,
                            static_groups=self.static_groups,
                            sym=self.sym,
                            percdamp=self.percdamp,
                            preproc_hessian=self.preproc_hessian,
                            hessian_device=torch.device("cpu"),
                            rtn=self._expert_rtn,
                            ssz=self._expert_ssz,
                            ssz_step=self.ssz_step,
                            clip_search=self._clip_cfg,
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
                    wbits=self._resolve_layer_wbits(module_rel_name),
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
    _NPU_SUFFIXES = {"weight", "weight_scale", "weight_offset", "scale_bias"}
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
                                wbits=self._resolve_layer_wbits(expert_rel_name),
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
                    wbits=self._resolve_layer_wbits(module_rel_name),
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
            model_quant_type = self._ascend_quant_type
            if self.layer_routing:
                # Mixed quant (e.g. MoE W4A8 + dense W8A8): top-level model_quant_type
                # is W8A8_DYNAMIC (verified against GLM-5.2-w4a8c8 artifact). Per-tensor
                # annotation in the desc drives per-layer scheme selection.
                route_wbits = {int(e.get("wbits", self.wbits)) for e in self.layer_routing}
                max_wbits = max(route_wbits | {self.wbits})
                model_quant_type = (
                    f"W{max_wbits}A8_DYNAMIC" if self.act_bits == 8 else f"W{max_wbits}A16"
                )
            self._model_config.ascend_quant_config = {
                "model_quant_type": model_quant_type,
                "group_size": self.groupsize,
                "quant_layer_types": ["GPTQQuantLinear"],
                "include_g_idx": True,
                "has_offset": True,
                "act_bits": self.act_bits,
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
