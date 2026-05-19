"""Chunk-wise SparseGPT pruning algorithm for compressor task.

Design:
- SparseGPTAlgorithm uses the shared Hessian runtime-model lifecycle,
  calibration capture, tensor management, and progressive-forward
  infrastructure from BaseHessianAlgorithm.
- Only the layer-level optimisation differs: in-place pruning via
  :class:`SparseGPTModule` instead of quantize-and-pack.
- SparseGPTModule inherits BaseHessianModule for Hessian accumulation/inversion.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from npuslim.algorithms.quantization.hessian import (
    BaseHessianAlgorithm,
    BaseHessianModule,
    _get_child_module,
    _is_transformers_conv1d,
)
from npuslim.core import AlgorithmRegistry


def _validate_sparsegpt_params(
    *,
    sparsity: float,
    prunen: int,
    prunem: int,
    blocksize: int,
    percdamp: float,
) -> None:
    if not math.isfinite(float(sparsity)):
        raise ValueError("[SparseGPT] `sparsity` must be a finite float.")
    if not math.isfinite(float(percdamp)):
        raise ValueError("[SparseGPT] `percdamp` must be a finite float.")
    if int(blocksize) <= 0:
        raise ValueError(f"[SparseGPT] `blocksize` must be > 0, got {blocksize}.")
    if not (0.0 <= float(percdamp) < 1.0):
        raise ValueError(
            f"[SparseGPT] `percdamp` must be in [0, 1), got {percdamp}."
        )

    prunen = int(prunen)
    prunem = int(prunem)
    if prunen < 0 or prunem < 0:
        raise ValueError(
            f"[SparseGPT] `prunen`/`prunem` must be >= 0, got {prunen}:{prunem}."
        )

    if prunen == 0:
        if not (0.0 <= float(sparsity) < 1.0):
            raise ValueError(
                f"[SparseGPT] unstructured mode requires `sparsity` in [0, 1), got {sparsity}."
            )
        return

    if prunem <= 0:
        raise ValueError(
            f"[SparseGPT] N:M mode requires `prunem` > 0 when `prunen` > 0, got {prunen}:{prunem}."
        )
    if prunen > prunem:
        raise ValueError(
            f"[SparseGPT] invalid N:M setting, require prunen <= prunem, got {prunen}:{prunem}."
        )

    if prunen > 0 and sparsity > 0:
        logger.warning(
            f"[SparseGPT] both `sparsity` ({sparsity}) and `prunen:prunem` "
            f"({prunen}:{prunem}) are set — N:M semi-structured mode takes "
            f"precedence; `sparsity` is ignored."
        )


# ---- 2:4 sparse packing utilities for Ascend NPU ----

_K_TILE_INDEX = 8


def _pack_sparse24(weight_int8: torch.Tensor):
    """Pack a 2:4 sparse int8 weight into densified values + tiled index.

    Args:
        weight_int8: [outfeatures, infeatures] int8 tensor with exact 2:4 sparsity.

    Returns:
        b_dense: [outfeatures, infeatures // 2] int8 tensor.
        index_tiled: 1D uint8 tensor in tiled format for AscendC.
    """
    N, K = weight_int8.shape
    assert K % 4 == 0, f"K must be divisible by 4, got {K}"
    num_groups = K // 4

    w = weight_int8.reshape(N, num_groups, 4)
    nz = w != 0
    nz_count = nz.sum(dim=-1)

    pos = torch.arange(4, device=w.device, dtype=torch.float32).reshape(1, 1, 4)
    pos_expanded = pos.expand(N, num_groups, 4).clone()
    pos_expanded[~nz] = 100.0
    sorted_pos, _ = pos_expanded.sort(dim=-1)

    first_pos = sorted_pos[:, :, 0].long().clamp(0, 3)
    second_pos = sorted_pos[:, :, 1].long().clamp(0, 3)

    idx1 = torch.zeros((N, num_groups), dtype=torch.long, device=w.device)
    idx2 = torch.zeros((N, num_groups), dtype=torch.long, device=w.device)
    val1 = torch.zeros((N, num_groups), dtype=torch.int8, device=w.device)
    val2 = torch.zeros((N, num_groups), dtype=torch.int8, device=w.device)

    # Exact 2:4 case: keep the first two non-zeros in ascending position order.
    multi_mask = nz_count >= 2
    if multi_mask.any():
        idx1[multi_mask] = first_pos[multi_mask]
        idx2[multi_mask] = (second_pos[multi_mask] - 1).clamp(0, 2)
        val1[multi_mask] = w.gather(2, first_pos.unsqueeze(-1)).squeeze(-1)[multi_mask]
        val2[multi_mask] = w.gather(2, second_pos.unsqueeze(-1)).squeeze(-1)[multi_mask]

    # Quantization can round one kept weight to zero, leaving only one non-zero
    # in a 4-tuple. Match the kernel-side reference encoding from the operator
    # test: keep the single value in slot 0 and encode the second slot with a
    # sentinel-compatible index.
    single_mask = nz_count == 1
    if single_mask.any():
        single_pos = first_pos[single_mask]
        single_val = w.gather(2, first_pos.unsqueeze(-1)).squeeze(-1)[single_mask]
        idx1[single_mask] = torch.where(single_pos < 3, single_pos, torch.zeros_like(single_pos))
        idx2[single_mask] = torch.where(single_pos < 3, torch.zeros_like(single_pos), torch.full_like(single_pos, 2))
        val1[single_mask] = single_val

    b_dense = torch.stack([val1, val2], dim=-1).reshape(N, K // 2)

    idx_pairs = torch.stack([idx1, idx2], dim=-1).reshape(N, num_groups * 2)
    idx_groups = idx_pairs.reshape(N, num_groups * 2 // 4, 4)
    shifts = torch.arange(4, device=w.device).reshape(1, 1, 4) * 2
    index_bytes = (idx_groups << shifts).sum(dim=-1).to(torch.uint8)

    index_tiled = _tile_index(index_bytes)
    return b_dense.contiguous(), index_tiled


def _tile_index(index_matrix: torch.Tensor, k_tile_index: int = _K_TILE_INDEX) -> torch.Tensor:
    """Convert [N, K//8] index matrix to tiled format for AscendC.

    Layout: each K_tile's index is stored as a contiguous [N_padded, k_tile_index]
    block, with blocks concatenated sequentially. N is padded to a multiple of 16.
    """
    N, K8 = index_matrix.shape
    pad_n = math.ceil(N / 16) * 16

    padded = torch.zeros(pad_n, K8, dtype=torch.uint8, device=index_matrix.device)
    padded[:N, :K8] = index_matrix

    num_tiles = math.ceil(K8 / k_tile_index)
    chunks = []
    for t in range(num_tiles):
        start = t * k_tile_index
        end = min(start + k_tile_index, K8)
        chunk = padded[:, start:end]
        if chunk.shape[1] < k_tile_index:
            chunk = torch.nn.functional.pad(chunk, (0, k_tile_index - chunk.shape[1]))
        chunks.append(chunk.reshape(-1))

    return torch.cat(chunks)


class AscendSparse24Linear(nn.Module):
    """2:4 structured sparse linear layer for Ascend NPU.

    Stores densified non-zero values, per-channel quantization scales,
    and tiled index for the AscendC sparse_matmul_4to2 operator.
    """

    def __init__(
        self,
        infeatures: int,
        outfeatures: int,
        bias: bool = False,
        weight_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.infeatures = int(infeatures)
        self.outfeatures = int(outfeatures)

        if self.infeatures % 4 != 0:
            raise ValueError(
                f"AscendSparse24Linear requires infeatures divisible by 4, got {self.infeatures}"
            )

        self.register_buffer(
            "weight",
            torch.zeros(self.outfeatures, self.infeatures // 2, dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(self.outfeatures, dtype=weight_dtype),
        )

        pad_n = math.ceil(self.outfeatures / 16) * 16
        k8 = self.infeatures // 8
        num_tiles = math.ceil(k8 / _K_TILE_INDEX)
        index_size = num_tiles * pad_n * _K_TILE_INDEX
        self.register_buffer(
            "weight_index",
            torch.zeros(index_size, dtype=torch.uint8),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(self.outfeatures, dtype=weight_dtype))
        else:
            self.bias = None

    @torch.no_grad()
    def pack(self, linear_weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Pack a 2:4 sparse weight into densified + tiled index format.

        Quantizes float weights to int8 (per-channel symmetric) before packing.

        Args:
            linear_weight: [outfeatures, infeatures] float tensor with 2:4 sparsity.
            bias: Optional bias tensor [outfeatures].
        """
        w = linear_weight.detach().float()

        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / 127.0
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)

        b_dense, index_tiled = _pack_sparse24(w_int8)

        self.weight.copy_(b_dense)
        self.weight_scale.copy_(scale.squeeze(1).to(self.weight_scale.dtype))
        self.weight_index.copy_(index_tiled)

        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.detach().to(dtype=self.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W^T via AscendC sparse matmul.

        Args:
            x: [..., infeatures] float tensor.

        Returns:
            [..., outfeatures] float tensor.
        """
        from npuslim.ops.sparse_matmul import sparse_matmul_4to2

        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.infeatures)

        max_val = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_scale = max_val / 127.0
        x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

        # [M, N] int32
        c_int32 = sparse_matmul_4to2(x_int8, self.weight, self.weight_index)

        # Dequantize: int32 -> float, absorbing both x_scale and weight_scale
        c_float = c_int32.float() * (x_scale * self.weight_scale.unsqueeze(0))
        out = c_float.to(x.dtype).reshape(*orig_shape, self.outfeatures)

        if self.bias is not None:
            out = out + self.bias
        return out


class SparseGPTModule(BaseHessianModule):
    """Per-linear SparseGPT pruning module.

    Reuses Hessian accumulation (``add_batch``) and inversion (``compute_hinv``)
    from :class:`BaseHessianModule`. Only ``fasterprune`` is SparseGPT-specific.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        sparsity: float = 0.0,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
    ):
        super().__init__(
            layer=layer,
            percdamp=percdamp,
            preproc_hessian=preproc_hessian,
        )
        self.sparsity = float(sparsity)
        self.prunen = int(prunen)
        self.prunem = int(prunem)
        self.blocksize = int(blocksize)

    @torch.no_grad()
    def fasterprune(self, **kwargs) -> Dict[str, float]:
        """Block-wise pruning with Hessian-weighted error correction."""
        _ = kwargs

        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if _is_transformers_conv1d(self.layer):
            W = W.t()

        W_orig = W.clone()
        H = self.H
        del self.H

        Hinv = self.compute_hinv(H)
        Losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if self.prunen == 0:
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * self.sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.prunen != 0 and i % self.prunem == 0:
                    window = min(self.prunem, count - i)
                    k = min(self.prunen, window)
                    tmp = (
                        W1[:, i : (i + window)] ** 2
                        / (torch.diag(Hinv1)[i : (i + window)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, k, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        avg_loss = torch.sum(Losses).item() / self.nsamples
        norm_loss = torch.norm(W - W_orig).item()

        if _is_transformers_conv1d(self.layer):
            W = W.t()

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.postproc()
        self.free()

        return {"rows": self.rows, "columns": self.columns, "avg_loss": avg_loss, "norm_loss": norm_loss}


class _SparseMode(Enum):
    """Platform x sparsity-type matrix for SparseGPT."""

    GPU_UNSTRUCTURED = auto()
    GPU_STRUCTURED = auto()
    NPU_UNSTRUCTURED = auto()
    NPU_STRUCTURED = auto()

    @staticmethod
    def resolve(target_backend: str, prunen: int, prunem: int) -> _SparseMode:
        on_npu = target_backend == "npu"
        structured = prunen > 0 and prunem > 0
        if on_npu and structured:
            return _SparseMode.NPU_STRUCTURED
        if on_npu:
            return _SparseMode.NPU_UNSTRUCTURED
        if structured:
            return _SparseMode.GPU_STRUCTURED
        return _SparseMode.GPU_UNSTRUCTURED


@AlgorithmRegistry.register("SparseGPT", aliases=["sparsegpt", "sparse_gpt"])
class SparseGPTAlgorithm(BaseHessianAlgorithm):
    """Chunk-wise SparseGPT pruning algorithm."""

    _TAG = "SparseGPT"
    _quantized_type_label = "SparseGPT"

    def __init__(
        self,
        sparsity: float = 0.0,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        fake_quant: bool = False,
        max_calib_samples: int = 128,
        save_backend: Optional[str] = None,
        **kwargs,
    ):
        _validate_sparsegpt_params(
            sparsity=float(sparsity),
            prunen=int(prunen),
            prunem=int(prunem),
            blocksize=int(blocksize),
            percdamp=float(percdamp),
        )
        super().__init__(max_calib_samples=max_calib_samples, **kwargs)
        self.sparsity = float(sparsity)
        self.prunen = int(prunen)
        self.prunem = int(prunem)
        self.blocksize = int(blocksize)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self.fake_quant = bool(fake_quant)
        self._save_backend = save_backend

        self._sparse_mode = _SparseMode.resolve(self.target_backend, self.prunen, self.prunem)
        logger.info(f"[{self._TAG}] {self._sparse_mode.name}, fake_quant={self.fake_quant}")

    @property
    def _use_sparse24_packing(self) -> bool:
        if self._sparse_mode is _SparseMode.NPU_STRUCTURED and not self.fake_quant:
            if not (self.prunen == 2 and self.prunem == 4):
                raise ValueError(
                    f"[{self._TAG}] only 2:4 structured sparsity is supported for "
                    f"sparse24 packing, got N:M={self.prunen}:{self.prunem}"
                )
            return True
        return False

    @property
    def _ascend_quant_type(self) -> str:
        return "Sparse24" if self._use_sparse24_packing else "FLOAT"

    def _log_start_params(self) -> None:
        sparsity_desc = (
            f"N:M={self.prunen}:{self.prunem}"
            if self.prunen > 0
            else f"unstructured={self.sparsity:.2f}"
        )
        logger.info(
            f"[{self._TAG}] start: {sparsity_desc}, "
            f"blocksize={self.blocksize}, percdamp={self.percdamp}, "
            f"sparse_mode={self._sparse_mode.name}, fake_quant={self.fake_quant}"
        )

    # ---- per-mode tensor saving ----

    def _save_fake_quant_weight(self, layer, handler, rel_weight_name, weight_tensor) -> str:
        """Save pruned weight as-is."""
        layer.tensors[rel_weight_name] = (
            handler.layer.weight.detach().to(weight_tensor.dtype).cpu()
        )
        return f"{layer.name}.{rel_weight_name}"

    def _save_sparse24_weight(self, layer, handler, module_rel_name, rel_weight_name, rel_bias_name) -> set[str]:
        """Pack pruned weight into 2:4 sparse format for Ascend NPU."""
        sparse_linear = AscendSparse24Linear(
            infeatures=handler.layer.in_features,
            outfeatures=handler.layer.out_features,
            bias=handler.layer.bias is not None,
            weight_dtype=handler.layer.weight.dtype,
        )
        bias_tensor = (
            handler.layer.bias.detach().cpu() if handler.layer.bias is not None else None
        )
        sparse_linear.pack(handler.layer.weight.detach().cpu(), bias=bias_tensor)

        layer.tensors.pop(rel_weight_name, None)
        layer.tensors.pop(rel_bias_name, None)

        packed_names: set[str] = set()
        for buf_name in ("weight", "weight_scale", "weight_index"):
            key = f"{module_rel_name}.{buf_name}"
            layer.tensors[key] = getattr(sparse_linear, buf_name).cpu()
            packed_names.add(f"{layer.name}.{key}")

        if sparse_linear.bias is not None:
            layer.tensors[f"{module_rel_name}.bias"] = sparse_linear.bias.cpu()

        return packed_names

    # ---- BaseHessianAlgorithm hooks ----

    def _create_handlers(self, layer_module: nn.Module, targets) -> Dict[str, SparseGPTModule]:
        handlers: Dict[str, SparseGPTModule] = {}
        for module_rel_name, *_rest in targets:
            submodule = _get_child_module(layer_module, module_rel_name)
            if not (isinstance(submodule, nn.Linear) or _is_transformers_conv1d(submodule)):
                continue
            handlers[module_rel_name] = SparseGPTModule(
                submodule,
                sparsity=self.sparsity,
                prunen=self.prunen,
                prunem=self.prunem,
                blocksize=self.blocksize,
                percdamp=self.percdamp,
                preproc_hessian=self.preproc_hessian,
            )
        return handlers

    def _process_layer_handlers(self, layer, targets, handlers, chunk) -> tuple[set[str], int]:
        _ = chunk
        quantized_tensor_names: set[str] = set()
        pruned_weights = 0

        for module_rel_name, rel_weight_name, rel_bias_name, weight_tensor, _bias in targets:
            handler = handlers.get(module_rel_name)
            if handler is None:
                continue
            metrics = handler.fasterprune(layer_name=f"{layer.name}.{module_rel_name}")

            if self._use_sparse24_packing:
                names = self._save_sparse24_weight(
                    layer, handler, module_rel_name, rel_weight_name, rel_bias_name,
                )
                quantized_tensor_names.update(names)
            else:
                name = self._save_fake_quant_weight(layer, handler, rel_weight_name, weight_tensor)
                quantized_tensor_names.add(name)

            pruned_weights += 1
            full_name = f"{layer.name}.{module_rel_name}"
            logger.info(
                f"[{self._TAG}] {full_name:<50s} | "
                f"shape=[{int(metrics.get('rows', 0)):>5},{int(metrics.get('columns', 0)):>5}] | "
                f"avg_loss={float(metrics.get('avg_loss', 0.0)):<12.6f} | "
                f"norm_loss={float(metrics.get('norm_loss', 0.0)):<12.6f}"
            )
        return quantized_tensor_names, pruned_weights

    def _update_quantization_metadata(self) -> None:
        if not self._use_sparse24_packing:
            return
        if self._model_config is None:
            return

        self._model_config.ascend_quant_config = {
            "model_quant_type": "Sparse24",
            "quant_layer_types": ["AscendSparse24Linear"],
            "sparsity_type": "2:4",
        }
        if hasattr(self._model_config, "quantization_config"):
            try:
                delattr(self._model_config, "quantization_config")
            except Exception:
                pass

        self._mark_model_quantized()
        logger.info(f"[{self._TAG}] model quantization metadata updated")

    def _finalize_chunk_metadata(self, chunk, quantized_tensor_names: set[str]) -> None:
        if not quantized_tensor_names:
            return
        tensor_types = {name: "FLOAT" for name in chunk.all_tensors().keys()}
        quantized_type = self._ascend_quant_type if self.target_backend == "npu" else self._quantized_type_label
        for name in quantized_tensor_names:
            if name in tensor_types:
                tensor_types[name] = quantized_type
        chunk.metadata["tensor_types"] = tensor_types
