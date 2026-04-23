"""Chunk-wise GPTQ algorithm for v2 compressor task.

Design goal:
- Strict chunk lifecycle: calibrate -> quantize -> pack inside each process_chunk call.
- No full model load requirement.
- Keep GPTQ core math/packing behavior aligned with v1 implementation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers  # type: ignore
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from loguru import logger
from tqdm import tqdm

from npuslim.algorithms.quantization.base_quant_algo import BaseQuantizationAlgorithm
from npuslim.core.backend import bh
from npuslim.models.base_model import get_hub_class
from npuslim.core import register_algorithm


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


class BaseHessianModule:
    """Hessian accumulator shared by GPTQ modules."""

    def __init__(
        self,
        layer: nn.Module,
        *,
        percdamp: float = 0.01,
        preproc_hessian: bool = False,
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        w = layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if _is_transformers_conv1d(self.layer):
            w = w.t()

        self.rows = w.shape[0]
        self.columns = w.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.preproc_done = False

        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)

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
        self.H += inp.matmul(inp.t())

    def compute_hinv(self, hessian: torch.Tensor) -> torch.Tensor:
        step = 0.01
        current_percdamp = 0.0 if self.preproc_hessian else float(self.percdamp)
        hinv = None

        while current_percdamp < 1.0:
            try:
                h_try = hessian.clone()
                if current_percdamp > 0:
                    damp = current_percdamp * torch.mean(torch.diag(h_try))
                    if damp == 0:
                        damp = 1e-5
                    diag_idx = torch.arange(self.columns, device=self.dev)
                    h_try[diag_idx, diag_idx] += damp

                if bh.name == "npu":
                    h_cpu = h_try.to("cpu")
                    l_cpu = torch.linalg.cholesky(h_cpu)
                    inv_l_cpu = torch.cholesky_inverse(l_cpu)
                    hinv_cpu = torch.linalg.cholesky(inv_l_cpu, upper=True)
                    hinv = hinv_cpu.to(self.dev)
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
            raise RuntimeError("Hessian inversion failed even with max damping.")
        return hinv

    def preproc(self) -> None:
        if self.preproc_hessian:
            w = self.layer.weight.data.clone()
            h = self.H.data.clone()
            dead = torch.diag(h) == 0
            h[dead, dead] = 1
            w[:, dead] = 0
            damp = self.percdamp * torch.mean(torch.diag(h))
            diag = torch.arange(self.columns, device=self.dev)
            h[diag, diag] += damp
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = h.to(self.H.data.dtype)
        self.preproc_done = True

    def postproc(self) -> None:
        assert self.preproc_done is True

    def free(self) -> None:
        self.H = None
        self.Losses = None
        self.Trace = None
        bh.empty_cache()


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
    ):
        super().__init__(
            layer=layer,
            percdamp=percdamp,
            preproc_hessian=preproc_hessian,
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

    def fasterquant(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = kwargs
        w_orig = self.layer.weight.data.float().clone()
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if _is_transformers_conv1d(self.layer):
            w = w.t()

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

        hessian = self.H
        if self.actorder:
            perm = torch.argsort(torch.diag(hessian), descending=True)
            w = w[:, perm]
            hessian = hessian[perm][:, perm]
            invperm = (
                torch.argsort(perm.float()) if bh.name == "npu" else torch.argsort(perm)
            )
        else:
            perm = torch.arange(self.columns, device=w.device)
            invperm = perm

        hinv = self.compute_hinv(hessian)
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
            "avg_loss": avg_loss,
            "norm_loss": norm_loss,
            "nsamples": float(self.nsamples),
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

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (weight[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
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


@register_algorithm("GPTQ", aliases=["gptq", "GPTQStepwise", "GPTQExample", "gptq_stepwise"])
class GPTQAlgorithm(BaseQuantizationAlgorithm):
    """Chunk-wise GPTQ algorithm."""

    _TAG = "GPTQ"

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
        **kwargs,
    ):
        if w_bits is not None:
            wbits = int(w_bits)
        if group_size is not None:
            groupsize = int(group_size)
        super().__init__(
            wbits=wbits,
            groupsize=groupsize,
            sym=sym,
            blocksize=blocksize,
            actorder=actorder,
            static_groups=static_groups,
            percdamp=percdamp,
            preproc_hessian=preproc_hessian,
            fake_quant=fake_quant,
            max_calib_samples=max_calib_samples,
            **kwargs,
        )
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.sym = bool(sym)
        self.blocksize = int(blocksize)
        self.actorder = bool(actorder)
        self.static_groups = bool(static_groups)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self.fake_quant = bool(fake_quant)
        self.max_calib_samples = max(int(max_calib_samples), 1)
        self._calib_batch_size = 1

        self._runtime_model: Optional[nn.Module] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_state_keys: set[str] = set()
        self._block_name: str = "model.layers"
        self._total_layers: int = 0
        self._next_expected_layer_index: Optional[int] = None
        self._inps: Optional[torch.Tensor] = None
        self._outs: Optional[torch.Tensor] = None
        self._layer_kwargs: Dict[str, Any] = {}

    @property
    def _ascend_quant_type(self) -> str:
        return f"W{self.wbits}A16"

    def on_start(self) -> None:
        logger.info(
            f"[{self._TAG}] start: "
            f"wbits={self.wbits}, groupsize={self.groupsize}, "
            f"actorder={self.actorder}, percdamp={self.percdamp}"
        )
        if self._model_obj is None:
            raise ValueError(f"[{self._TAG}] runtime model is required")

        self._runtime_model = self._build_runtime_model()
        runtime_cfg = getattr(self._runtime_model, "config", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "use_cache"):
            try:
                runtime_cfg.use_cache = False
            except Exception:
                pass
        self._runtime_state_keys = set(self._runtime_model.state_dict().keys())
        self._block_name = getattr(
            self._model_obj,
            "block_name",
            getattr(self._model_obj, "layers_path", "model.layers"),
        )
        layer_container = _get_child_module(self._runtime_model, self._block_name)
        self._total_layers = len(layer_container)
        self._runtime_device = None
        self._next_expected_layer_index = None
        self._inps = None
        self._outs = None
        self._layer_kwargs = {}
        self._calib_batch_size = 1

    def on_finish(self) -> None:
        self._update_quantization_metadata()
        self._runtime_model = None
        self._runtime_state_keys.clear()
        self._runtime_device = None
        self._inps = None
        self._outs = None
        self._layer_kwargs = {}
        self._next_expected_layer_index = None
        self._calib_batch_size = 1
        if self._model_obj is not None and hasattr(self._model_obj, "release_empty_model"):
            self._model_obj.release_empty_model()
        bh.empty_cache()
        logger.info(f"[{self._TAG}] finish")

    def _build_runtime_model(self) -> nn.Module:
        if self._model_obj is not None and hasattr(self._model_obj, "prepare_empty_model"):
            model = self._model_obj.prepare_empty_model()
            if model is not None:
                model.eval()
                return model

        if self._model_config is None:
            raise ValueError(f"[{self._TAG}] model_config is required to build empty runtime model")

        auto_model_cls = get_hub_class(
            getattr(self._model_obj, "model_hub", "hf"),
            "AutoModelForCausalLM",
        )
        model_kwargs = dict(getattr(self._model_obj, "model_kwargs", {}) or {})
        trust_remote_code = bool(model_kwargs.get("trust_remote_code", False))
        extra_kwargs: Dict[str, Any] = {}
        if "attn_implementation" in model_kwargs:
            extra_kwargs["attn_implementation"] = model_kwargs["attn_implementation"]
        if "torch_dtype" in model_kwargs:
            extra_kwargs["torch_dtype"] = model_kwargs["torch_dtype"]

        with init_empty_weights():
            model = auto_model_cls.from_config(
                self._model_config,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        model.eval()
        return model

    @staticmethod
    def _extract_linear_targets(layer, skip_names: List[str]):
        targets = []
        for rel_name, tensor in layer.tensors.items():
            if not rel_name.endswith(".weight"):
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_floating_point() or tensor.ndim != 2:
                continue
            module_rel_name = rel_name[:-7]
            full_weight_name = f"{layer.name}.{rel_name}"
            full_module_name = f"{layer.name}.{module_rel_name}"
            if GPTQAlgorithm.should_skip_name(full_weight_name, skip_names):
                continue
            if GPTQAlgorithm.should_skip_name(full_module_name, skip_names):
                continue
            bias_name = f"{module_rel_name}.bias"
            bias = layer.tensors.get(bias_name)
            targets.append((module_rel_name, rel_name, bias_name, tensor, bias))
        return targets

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
            backend=bh.name,
        )
        proxy = SimpleNamespace(
            weight=linear_module.weight.detach().cpu(),
            bias=(linear_module.bias.detach().cpu() if linear_module.bias is not None else None),
        )
        quant_linear.pack(proxy, scales.cpu(), zeros.cpu(), g_idx.cpu())

        tensors: Dict[str, torch.Tensor] = {}
        quantized_names: List[str] = []
        if bh.name == "npu":
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

    @staticmethod
    def _full_tensor_name(module_name: str, rel_name: str) -> str:
        if rel_name.startswith(f"{module_name}."):
            return rel_name
        return f"{module_name}.{rel_name}"

    def _collect_module_runtime_tensors(
        self,
        module_name: str,
        module_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        tensor_map: Dict[str, torch.Tensor] = {}
        for rel_name, tensor in module_tensors.items():
            if not torch.is_tensor(tensor):
                continue
            full_name = self._full_tensor_name(module_name, rel_name)
            if full_name in self._runtime_state_keys:
                tensor_map[full_name] = tensor
        return tensor_map

    def _resolve_runtime_device(self, chunk) -> torch.device:
        for tensor in chunk.all_tensors().values():
            if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                return tensor.device
        backend_device = getattr(bh, "device", None)
        if isinstance(backend_device, torch.device):
            return backend_device
        return torch.device(bh.default_device_str())

    def _assign_runtime_tensors(self, tensor_map: Dict[str, torch.Tensor]) -> List[str]:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] runtime model is not initialized")
        if self._runtime_device is None:
            raise RuntimeError(f"[{self._TAG}] runtime device is not initialized")

        assigned: List[str] = []
        for full_name, tensor in tensor_map.items():
            set_module_tensor_to_device(
                self._runtime_model,
                full_name,
                device=self._runtime_device,
                value=tensor if tensor.device == self._runtime_device else tensor.to(self._runtime_device),
            )
            assigned.append(full_name)
        return assigned

    def _unassign_runtime_tensors(self, tensor_names: List[str]) -> None:
        if self._runtime_model is None:
            return
        for full_name in tensor_names:
            module_name, leaf_name = full_name.rsplit(".", 1)
            module = _get_child_module(self._runtime_model, module_name)
            current = getattr(module, leaf_name)
            meta_tensor = torch.empty_like(current, device="meta")
            set_module_tensor_to_device(
                self._runtime_model,
                full_name,
                device="meta",
                value=meta_tensor,
            )

    @staticmethod
    def _iter_calib_batches(calib_data: Any):
        if calib_data is None:
            return
        for batch in calib_data:
            yield batch

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self._runtime_device is None:
            raise RuntimeError(f"[{self._TAG}] runtime device is not initialized")
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self._runtime_device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _sanitize_layer_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize per-layer kwargs to avoid cache growth during calibration."""
        sanitized = dict(kwargs)
        # Match v1 behavior: calibration path must not build/update KV cache.
        sanitized["use_cache"] = False
        if "past_key_values" in sanitized:
            sanitized["past_key_values"] = None
        return sanitized

    def _capture_initial_inputs(self, chunk) -> None:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] runtime model is not initialized")
        if not chunk.layers:
            raise ValueError(f"[{self._TAG}] first chunk must contain at least one layer")
        if chunk.calib_data is None:
            raise ValueError(f"[{self._TAG}] calibration data is required")

        class _CaptureInputsStop(RuntimeError):
            pass

        first_layer = _get_child_module(self._runtime_model, chunk.layers[0].name)
        captured: List[torch.Tensor] = []
        layer_kwargs: Dict[str, Any] = {}

        def hook_fn(module, args, kwargs):
            _ = module
            if not args:
                raise _CaptureInputsStop()
            captured.append(args[0].detach())
            if not layer_kwargs:
                for key, value in kwargs.items():
                    if torch.is_tensor(value):
                        layer_kwargs[key] = value.detach()
                    else:
                        layer_kwargs[key] = value
            raise _CaptureInputsStop()

        handle = first_layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        sample_count = 0
        total_batches = len(chunk.calib_data) if hasattr(chunk.calib_data, "__len__") else None
        try:
            with torch.no_grad():
                batch_iter = tqdm(
                    self._iter_calib_batches(chunk.calib_data),
                    total=total_batches,
                    desc=f"{self._TAG.lower()} capture first-layer",
                    leave=True,
                )
                for raw_batch in batch_iter:
                    if not isinstance(raw_batch, dict):
                        continue
                    batch_no_label = dict(raw_batch)
                    batch_no_label.pop("labels", None)
                    batch = self._move_batch_to_device(batch_no_label)
                    try:
                        self._runtime_model(**batch)
                    except _CaptureInputsStop:
                        pass
                    if captured:
                        sample_count += int(captured[-1].shape[0])
                        if sample_count >= self.max_calib_samples:
                            break
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"[{self._TAG}] failed to capture first-layer inputs from calibration data")

        inps = torch.cat(captured, dim=0)
        inps = inps[: self.max_calib_samples]
        self._inps = inps
        self._outs = torch.zeros_like(inps)
        self._layer_kwargs = self._sanitize_layer_kwargs(layer_kwargs)
        self._calib_batch_size = max(int(captured[0].shape[0]), 1)

    def _iter_layer_sample_ranges(self, total_samples: int):
        step = max(int(self._calib_batch_size), 1)
        for start in range(0, total_samples, step):
            end = min(start + step, total_samples)
            yield start, end

    def _collect_statistics(
        self,
        layer_module: nn.Module,
        handlers: Dict[str, GPTQModule],
        *,
        layer_name: str,
        chunk_index: int,
    ) -> None:
        if self._inps is None:
            raise RuntimeError(f"[{self._TAG}] missing captured inputs")
        hooks = []
        for handler in handlers.values():
            hooks.append(
                handler.layer.register_forward_hook(
                    lambda module, inp, out, h=handler: h.add_batch(inp[0].data, _unwrap_output(out).data)
                )
            )
        total_samples = int(self._inps.shape[0])
        step = max(int(self._calib_batch_size), 1)
        total_steps = (total_samples + step - 1) // step
        with torch.no_grad():
            sample_iter = tqdm(
                self._iter_layer_sample_ranges(total_samples),
                total=total_steps,
                desc=f"{self._TAG.lower()} calib c{chunk_index} {layer_name}",
                leave=True,
                disable=total_steps <= 1,
            )
            for start, end in sample_iter:
                layer_module(self._inps[start:end], **self._layer_kwargs)
        for hook in hooks:
            hook.remove()
        for handler in handlers.values():
            handler.preproc()

    def _forward_layer_outputs(
        self,
        layer_module: nn.Module,
        *,
        layer_name: str,
        chunk_index: int,
    ) -> None:
        if self._inps is None or self._outs is None:
            raise RuntimeError(f"[{self._TAG}] missing captured inputs")
        total_samples = int(self._inps.shape[0])
        step = max(int(self._calib_batch_size), 1)
        total_steps = (total_samples + step - 1) // step
        with torch.no_grad():
            sample_iter = tqdm(
                self._iter_layer_sample_ranges(total_samples),
                total=total_steps,
                desc=f"{self._TAG.lower()} forward c{chunk_index} {layer_name}",
                leave=True,
                disable=total_steps <= 1,
            )
            for start, end in sample_iter:
                out = layer_module(self._inps[start:end], **self._layer_kwargs)
                self._outs[start:end] = _unwrap_output(out).detach()
        self._inps, self._outs = self._outs, self._inps

    def _validate_chunk_order(self, chunk) -> None:
        if not chunk.layers:
            return
        indices = chunk.layer_indices
        if self._next_expected_layer_index is None:
            self._next_expected_layer_index = indices[0]
        expected = self._next_expected_layer_index
        for idx in indices:
            if idx != expected:
                raise ValueError(
                    f"[{self._TAG}] chunk order mismatch: expected layer {expected}, got {idx}"
                )
            expected += 1
        self._next_expected_layer_index = expected

    def _update_quantization_metadata(self) -> None:
        if self._model_config is None:
            return

        if bh.name == "npu":
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

    def process_chunk(self, chunk) -> Any:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] on_start must be called before process_chunk")
        self._validate_chunk_order(chunk)
        if self._runtime_device is None:
            self._runtime_device = self._resolve_runtime_device(chunk)

        skip_names = self._set_skip_from_chunk_metadata(chunk)

        quantized_tensor_names: set[str] = set()
        quantized_weights = 0

        if self._inps is None:
            if not chunk.is_first_chunk:
                raise ValueError(
                    f"[{self._TAG}] first processed chunk must include layer-0 to capture initial inputs"
                )
            pre_tensor_map: Dict[str, torch.Tensor] = {}
            for module in chunk.pre_modules:
                pre_tensor_map.update(
                    self._collect_module_runtime_tensors(module.name, module.tensors)
                )
            pre_assigned = self._assign_runtime_tensors(pre_tensor_map)
            try:
                self._capture_initial_inputs(chunk)
            finally:
                self._unassign_runtime_tensors(pre_assigned)

        layer_iter = tqdm(
            chunk.layers,
            total=chunk.layer_count,
            desc=f"{self._TAG.lower()} chunk {chunk.chunk_index}",
            leave=True,
            disable=chunk.layer_count <= 1,
        )
        for layer in layer_iter:
            layer_tensor_map = self._collect_module_runtime_tensors(layer.name, layer.tensors)
            layer_assigned = self._assign_runtime_tensors(layer_tensor_map)
            try:
                layer_module = _get_child_module(self._runtime_model, layer.name)
                targets = self._extract_linear_targets(layer, skip_names)
                handlers: Dict[str, GPTQModule] = {}
                for module_rel_name, *_ in targets:
                    submodule = _get_child_module(layer_module, module_rel_name)
                    if not isinstance(submodule, nn.Linear):
                        continue
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
                    )

                if handlers:
                    self._collect_statistics(
                        layer_module,
                        handlers,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )

                quant_results = []
                for (
                    module_rel_name,
                    rel_weight_name,
                    rel_bias_name,
                    weight_tensor,
                    _bias_tensor,
                ) in targets:
                    handler = handlers.get(module_rel_name)
                    if handler is None:
                        continue
                    scales, zeros, g_idx = handler.fasterquant(
                        layer_name=f"{layer.name}.{module_rel_name}"
                    )
                    metrics = getattr(handler, "last_metrics", {})
                    if metrics:
                        logger.info(
                            f"[{self._TAG}] layer={layer.name}.{module_rel_name} "
                            f"avg_loss={float(metrics.get('avg_loss', 0.0)):.6f} "
                            f"norm_loss={float(metrics.get('norm_loss', 0.0)):.6f}"
                        )
                    quant_results.append(
                        (
                            module_rel_name,
                            rel_weight_name,
                            rel_bias_name,
                            weight_tensor,
                            scales,
                            zeros,
                            g_idx,
                            handler,
                        )
                    )

                pack_iter = tqdm(
                    quant_results,
                    total=len(quant_results),
                    desc=f"{self._TAG.lower()} pack c{chunk.chunk_index} {layer.name}",
                    leave=True,
                    disable=len(quant_results) <= 1,
                )
                for (
                    module_rel_name,
                    rel_weight_name,
                    rel_bias_name,
                    weight_tensor,
                    scales,
                    zeros,
                    g_idx,
                    handler,
                ) in pack_iter:
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

                for handler in handlers.values():
                    handler.free()

                if layer.index < self._total_layers - 1:
                    self._forward_layer_outputs(
                        layer_module,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )
            finally:
                self._unassign_runtime_tensors(layer_assigned)

        tensor_types = {name: "FLOAT" for name in chunk.all_tensors().keys()}
        quantized_type = self._ascend_quant_type if bh.name == "npu" else "GPTQ"
        for name in quantized_tensor_names:
            if name in tensor_types:
                tensor_types[name] = quantized_type
        chunk.metadata["tensor_types"] = tensor_types

        logger.info(
            f"[{self._TAG}] chunk={chunk.chunk_index}, layers={chunk.layer_count}, "
            f"quantized_weights={quantized_weights}"
        )
        return chunk
