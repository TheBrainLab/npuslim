"""Shared utilities and Hessian accumulator for Hessian-based quantization algorithms.

Extracted from gptq_algo.py as the canonical source. Used by GPTQ, QuIP, and SparseGPT.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from loguru import logger

import torch
import torch.nn as nn
import transformers  # type: ignore

from npuslim.core.backend import bh


def _is_transformers_conv1d(layer: nn.Module) -> bool:
    conv1d_cls = getattr(transformers, "Conv1D", None)
    return conv1d_cls is not None and isinstance(layer, conv1d_cls)


def _unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


def _get_child_module(root: Any, dotted_name: str) -> Any:
    module = root
    if not dotted_name:
        return module
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


@torch.no_grad()
def compute_scales_with_zero(
    x: torch.Tensor,
    bits: int = 4,
    sym: bool = True,
    perchannel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        negative = xmin < 0
        if torch.any(negative):
            xmin[negative] = -xmax[negative]

    both_zero = (xmin == 0) & (xmax == 0)
    xmin[both_zero] = -1
    xmax[both_zero] = +1

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
    maxq = 2**bits - 1
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


# NOTE: Currently unused — kept as a fallback for NPU cholesky_inverse.
# See compute_hinv() for the active NPU path, which offloads to CPU instead.
def npu_cholesky_inverse(L: torch.Tensor, upper: bool = False) -> torch.Tensor:
    n = L.size(-1)
    eye = torch.eye(n, dtype=L.dtype, device=L.device)

    if upper:
        # U^T U = A  =>  solve U^T Y = I, then solve U X = Y
        y = torch.linalg.solve_triangular(L.transpose(-1, -2), eye, upper=False)
        x = torch.linalg.solve_triangular(L, y, upper=True)
    else:
        # L L^T = A  =>  solve L Y = I, then solve L^T X = Y
        y = torch.linalg.solve_triangular(L, eye, upper=False)
        x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)

    return x


class BaseHessianModule:
    """Hessian accumulator shared by GPTQ, QuIP, and SparseGPT modules."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        percdamp: float = 0.01,
        preproc_hessian: bool = False,
        hessian_device: Optional[torch.device] = None,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        self._hdev = hessian_device if hessian_device is not None else self.dev
        w = layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if _is_transformers_conv1d(self.layer):
            w = w.t()

        self.rows = w.shape[0]
        self.columns = w.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self._hdev)
        self.nsamples = 0
        self.preproc_done = False

        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self._dead_mask: Optional[torch.Tensor] = None
        self._hinv_fallback: Optional[str] = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        _ = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or _is_transformers_conv1d(self.layer):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        inp = inp.to(self.dev, dtype=torch.float32)
        self.H *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        inp = math.sqrt(2 / self.nsamples) * inp
        contrib = inp.matmul(inp.t())
        self.H += contrib.to(self._hdev)

    def compute_hinv(self, hessian: torch.Tensor) -> torch.Tensor:
        hessian = hessian.to(self.dev)
        step = 0.1
        current_percdamp = 0.0 if self.preproc_hessian else float(self.percdamp)
        hinv = None

        while current_percdamp < 10.0:
            try:
                h_try = hessian.clone()
                if current_percdamp > 0:
                    damp = current_percdamp * torch.mean(torch.diag(h_try))
                    if damp.item() == 0:
                        damp = 1e-5
                    diag_idx = torch.arange(self.columns, device=self.dev)
                    h_try[diag_idx, diag_idx] += damp

                if bh.has_npu:
                    h_cpu = h_try.to("cpu")
                    l_cpu = torch.linalg.cholesky(h_cpu)
                    inv_l_cpu = torch.cholesky_inverse(l_cpu)
                    hinv_cpu = torch.linalg.cholesky(inv_l_cpu, upper=True)
                    hinv = hinv_cpu.to(self.dev)
                    # chol = torch.linalg.cholesky(h_try)
                    # inv_chol = npu_cholesky_inverse(chol)
                    # hinv = torch.linalg.cholesky(inv_chol, upper=True)
                else:
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
            logger.warning(f"[GPTQ] Cholesky failed after max damping, using pseudo-inverse")
            self._hinv_fallback = "pinv"
            try:
                hinv = torch.linalg.pinv(hessian.to("cpu")).to(self.dev)
            except Exception:
                logger.warning(f"[GPTQ] Pseudo-inverse also failed, using identity (no quantization)")
                self._hinv_fallback = "identity"
                hinv = torch.eye(self.columns, device=self.dev)
        else:
            self._hinv_fallback = None
        return hinv

    def preproc(self) -> None:
        if self.preproc_hessian:
            h = self.H.data.clone()
            dead = torch.diag(h) == 0
            h[dead, dead] = 1
            damp = self.percdamp * torch.mean(torch.diag(h))
            diag = torch.arange(self.columns, device=self._hdev)
            h[diag, diag] += damp
            self.H.data = h.to(self.H.data.dtype)
            # Store dead mask for fasterquant to apply; do NOT mutate the
            # original weight tensor so failed quantizers can fall back cleanly.
            self._dead_mask = dead.to(self.dev)
        self.preproc_done = True

    def postproc(self) -> None:
        assert self.preproc_done is True

    def free(self) -> None:
        self.H = None
        self.Losses = None
        self.Trace = None
        bh.empty_cache()
