"""Chunk-wise SparseGPT pruning algorithm for v2 compressor task.

Design:
- SparseGPTAlgorithm inherits GPTQAlgorithm to reuse the entire runtime-model
  lifecycle, calibration capture, tensor management, and progressive-forward
  infrastructure.
- Only the layer-level optimisation differs: in-place pruning via
  :class:`SparseGPTModule` instead of quantize-and-pack.
- SparseGPTModule inherits BaseHessianModule for Hessian accumulation/inversion.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from npuslim.algorithms.quantization.gptq.gptq_algo import (
    BaseHessianModule,
    GPTQAlgorithm,
    _get_child_module,
    _is_transformers_conv1d,
)
from npuslim.registry import register_algorithm


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


# ---------------------------------------------------------------------------
# Per-layer pruning module
# ---------------------------------------------------------------------------


class SparseGPTModule(BaseHessianModule):
    """Per-linear SparseGPT pruning module.

    Reuses Hessian accumulation (``add_batch``) and inversion (``compute_hinv``)
    from :class:`BaseHessianModule`.  Only ``fasterprune`` is SparseGPT-specific.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        sparsity: float = 0.5,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
    ):
        _validate_sparsegpt_params(
            sparsity=float(sparsity),
            prunen=int(prunen),
            prunem=int(prunem),
            blocksize=int(blocksize),
            percdamp=float(percdamp),
        )
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

            # Mask computation
            if self.prunen == 0:
                # Mode A: unstructured sparsity
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * self.sparsity)]
                mask1 = tmp <= thresh
            else:
                # Mode B: N:M semi-structured sparsity
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.prunen != 0 and i % self.prunem == 0:
                    window = min(self.prunem, count - i)
                    k = min(self.prunen, window)
                    tmp = (
                        W1[:, i : (i + window)] ** 2
                        / (torch.diag(Hinv1)[i : (i + window)].reshape((1, -1)))
                        ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, k, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
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

        return {"avg_loss": avg_loss, "norm_loss": norm_loss}


# ---------------------------------------------------------------------------
# Top-level algorithm — inherits GPTQ's runtime infrastructure
# ---------------------------------------------------------------------------


@register_algorithm("SparseGPT", aliases=["sparsegpt", "sparse_gpt"])
class SparseGPTAlgorithm(GPTQAlgorithm):
    """Chunk-wise SparseGPT pruning algorithm.

    Inherits the full runtime-model lifecycle from :class:`GPTQAlgorithm` and
    overrides only the layer-level optimisation (pruning vs quantize-and-pack).
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        max_calib_samples: int = 128,
        **kwargs,
    ):
        _validate_sparsegpt_params(
            sparsity=float(sparsity),
            prunen=int(prunen),
            prunem=int(prunem),
            blocksize=int(blocksize),
            percdamp=float(percdamp),
        )
        # Pass common params to GPTQAlgorithm; GPTQ-specific ones get defaults.
        super().__init__(
            wbits=4,
            groupsize=128,
            sym=True,
            blocksize=blocksize,
            actorder=False,
            static_groups=True,
            percdamp=percdamp,
            preproc_hessian=preproc_hessian,
            fake_quant=False,
            max_calib_samples=max_calib_samples,
            **kwargs,
        )
        # SparseGPT-specific params
        self.sparsity = float(sparsity)
        self.prunen = int(prunen)
        self.prunem = int(prunem)

    # -- tag for log messages -------------------------------------------------

    _TAG = "SparseGPT"

    # -- lifecycle overrides ---------------------------------------------------

    def on_start(self) -> None:
        sparsity_desc = (
            f"N:M={self.prunen}:{self.prunem}"
            if self.prunen > 0
            else f"unstructured={self.sparsity:.2f}"
        )
        logger.info(
            f"[{self._TAG}] start: {sparsity_desc}, "
            f"blocksize={self.blocksize}, percdamp={self.percdamp}"
        )
        # Reuse GPTQ's on_start (runtime model init, input capture setup)
        GPTQAlgorithm.on_start(self)

    def on_finish(self) -> None:
        # Skip GPTQ's _update_quantization_metadata — pruning has no quant config
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
        from npuslim.core.backend import bh
        bh.empty_cache()
        logger.info(f"[{self._TAG}] finish")

    # -- main entry ------------------------------------------------------------

    def process_chunk(self, chunk):  # noqa: ANN201
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] on_start must be called before process_chunk")
        self._validate_chunk_order(chunk)
        if self._runtime_device is None:
            self._runtime_device = self._resolve_runtime_device(chunk)

        skip_names = self._set_skip_from_chunk_metadata(chunk)
        pruned_weights = 0

        if self._inps is None:
            if not chunk.is_first_chunk:
                raise ValueError(
                    f"[{self._TAG}] first processed chunk must include layer-0"
                )
            pre_tensor_map = {}
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
                handlers: Dict[str, SparseGPTModule] = {}
                for module_rel_name, *_rest in targets:
                    submodule = _get_child_module(layer_module, module_rel_name)
                    if not (
                        isinstance(submodule, nn.Linear)
                        or _is_transformers_conv1d(submodule)
                    ):
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

                if handlers:
                    self._collect_statistics(
                        layer_module,
                        handlers,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )

                for module_rel_name, rel_weight_name, _bias_name, weight_tensor, _bias in targets:
                    handler = handlers.get(module_rel_name)
                    if handler is None:
                        continue
                    metrics = handler.fasterprune(
                        layer_name=f"{layer.name}.{module_rel_name}"
                    )
                    layer.tensors[rel_weight_name] = (
                        handler.layer.weight.detach().to(weight_tensor.dtype).cpu()
                    )
                    pruned_weights += 1

                    logger.info(
                        f"[{self._TAG}] layer={layer.name}.{module_rel_name} "
                        f"avg_loss={float(metrics.get('avg_loss', 0.0)):.6f} "
                        f"norm_loss={float(metrics.get('norm_loss', 0.0)):.6f}",
                    )

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

        logger.info(
            f"[{self._TAG}] chunk={chunk.chunk_index}, "
            f"layers={chunk.layer_count}, pruned_weights={pruned_weights}"
        )
        return chunk
