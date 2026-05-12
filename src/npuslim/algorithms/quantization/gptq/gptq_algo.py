"""Chunk-wise GPTQ algorithm for compressor task.

Design goal:
- Strict chunk lifecycle: calibrate -> quantize -> pack inside each process_chunk call.
- No full model load requirement.
- Keep GPTQ core math/packing behavior aligned with v1 implementation.
"""

from __future__ import annotations

import math
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
from npuslim.core import register_algorithm
from npuslim.core.backend import bh


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
        **kwargs,
    ):
        if w_bits is not None:
            wbits = int(w_bits)
        if group_size is not None:
            groupsize = int(group_size)
        super().__init__(max_calib_samples=max_calib_samples, **kwargs)
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.sym = bool(sym)
        self.blocksize = int(blocksize)
        self.actorder = bool(actorder)
        self.static_groups = bool(static_groups)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self.fake_quant = bool(fake_quant)

    @property
    def _ascend_quant_type(self) -> str:
        return f"W{self.wbits}A16"

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

    def _create_handlers(self, layer_module: nn.Module, targets) -> Dict[str, GPTQModule]:
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
        return handlers

    def _process_layer_handlers(self, layer, targets, handlers, chunk) -> tuple[set[str], int]:
        quantized_tensor_names: set[str] = set()
        quantized_weights = 0
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
        return quantized_tensor_names, quantized_weights

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
