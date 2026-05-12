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
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger

from npuslim.algorithms.quantization.hessian import (
    BaseHessianAlgorithm,
    BaseHessianModule,
    _get_child_module,
    _is_transformers_conv1d,
)
from npuslim.core import register_algorithm


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


class SparseGPTModule(BaseHessianModule):
    """Per-linear SparseGPT pruning module.

    Reuses Hessian accumulation (``add_batch``) and inversion (``compute_hinv``)
    from :class:`BaseHessianModule`. Only ``fasterprune`` is SparseGPT-specific.
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

        return {"avg_loss": avg_loss, "norm_loss": norm_loss}


@register_algorithm("SparseGPT", aliases=["sparsegpt", "sparse_gpt"])
class SparseGPTAlgorithm(BaseHessianAlgorithm):
    """Chunk-wise SparseGPT pruning algorithm."""

    _TAG = "SparseGPT"
    _quantized_type_label = "SparseGPT"

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
        super().__init__(max_calib_samples=max_calib_samples, **kwargs)
        self.sparsity = float(sparsity)
        self.prunen = int(prunen)
        self.prunem = int(prunem)
        self.blocksize = int(blocksize)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)

    def _log_start_params(self) -> None:
        sparsity_desc = (
            f"N:M={self.prunen}:{self.prunem}"
            if self.prunen > 0
            else f"unstructured={self.sparsity:.2f}"
        )
        logger.info(
            f"[{self._TAG}] start: {sparsity_desc}, "
            f"blocksize={self.blocksize}, percdamp={self.percdamp}"
        )

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
        pruned_weights = 0
        for module_rel_name, rel_weight_name, _bias_name, weight_tensor, _bias in targets:
            handler = handlers.get(module_rel_name)
            if handler is None:
                continue
            metrics = handler.fasterprune(layer_name=f"{layer.name}.{module_rel_name}")
            layer.tensors[rel_weight_name] = (
                handler.layer.weight.detach().to(weight_tensor.dtype).cpu()
            )
            pruned_weights += 1
            logger.info(
                f"[{self._TAG}] layer={layer.name}.{module_rel_name} "
                f"avg_loss={float(metrics.get('avg_loss', 0.0)):.6f} "
                f"norm_loss={float(metrics.get('norm_loss', 0.0)):.6f}"
            )
        return set(), pruned_weights

    def _update_quantization_metadata(self) -> None:
        pass

    def _finalize_chunk_metadata(self, chunk, quantized_tensor_names: set[str]) -> None:
        _ = chunk
        _ = quantized_tensor_names
        pass
