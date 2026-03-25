"""INT8 Dynamic quantization algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from loguru import logger

from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.registry import register_algorithm

if TYPE_CHECKING:
    from npuslim.tasks.compressor.context import ChunkContext


@dataclass
class INT8DynamicConfig:
    wbits: int = 8
    w_quant_method: str = "per-channel"  # per-tensor / per-channel / per-group
    a_quant_method: str = "per-token"
    group_size: int = -1


class WeightObserver:
    """Weight observer with per-tensor/per-channel/per-group statistics."""

    def __init__(self, quant_bits: int, method: str, group_size: int = -1):
        self.quant_bits = int(quant_bits)
        self.method = str(method)
        self.group_size = int(group_size)
        self._scale: Optional[torch.Tensor] = None

    def _qmax(self) -> int:
        return (1 << (self.quant_bits - 1)) - 1

    def observe(self, weight: torch.Tensor) -> torch.Tensor:
        qmax = max(self._qmax(), 1)
        method = self.method

        if method == "per-group" and self.group_size > 0 and weight.ndim == 2:
            _, in_features = weight.shape
            g = self.group_size
            groups = (in_features + g - 1) // g
            scales: List[torch.Tensor] = []
            for idx in range(groups):
                s = idx * g
                e = min((idx + 1) * g, in_features)
                max_abs = weight[:, s:e].abs().amax(dim=1, keepdim=True)
                scale = max_abs / qmax
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                scales.append(scale)
            self._scale = torch.cat(scales, dim=1)
            return self._scale

        if method == "per-channel" and weight.ndim >= 2:
            max_abs = weight.abs().amax(dim=1, keepdim=True)
        else:
            max_abs = weight.abs().amax()

        scale = max_abs / qmax
        if isinstance(scale, torch.Tensor):
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self._scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, device=weight.device)
        return self._scale

    def scales(self) -> Optional[torch.Tensor]:
        return self._scale


class INT8LayerHandler:
    """Per-weight handler for quantization."""

    def __init__(self, wbits: int, method: str, group_size: int):
        self.wbits = int(wbits)
        self.method = str(method)
        self.group_size = int(group_size)
        self.observer = WeightObserver(
            quant_bits=wbits,
            method=method,
            group_size=group_size,
        )

    def _int_range(self) -> Tuple[int, int]:
        qmin = -(1 << (self.wbits - 1))
        qmax = (1 << (self.wbits - 1)) - 1
        return qmin, qmax

    def _qdq_with_scale(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        qmin, qmax = self._int_range()
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q * scale

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        scale = self.observer.scales()
        if scale is None:
            scale = self.observer.observe(weight)

        if self.method == "per-group" and self.group_size > 0 and weight.ndim == 2:
            _, in_features = weight.shape
            g = self.group_size
            groups = (in_features + g - 1) // g
            out = weight.clone()
            for idx in range(groups):
                s = idx * g
                e = min((idx + 1) * g, in_features)
                seg_scale = scale[:, idx:idx + 1]
                out[:, s:e] = self._qdq_with_scale(weight[:, s:e], seg_scale)
            return out.to(dtype=weight.dtype)

        return self._qdq_with_scale(weight, scale).to(dtype=weight.dtype)


@register_algorithm("INT8Dynamic", aliases=["INT8Dyn", "int8_dyn"])
class INT8DynamicAlgorithm(BaseAlgorithm):
    """INT8 dynamic quantization with observer/handler workflow."""

    def __init__(
        self,
        wbits: int = 8,
        w_quant_method: str = "per-channel",
        a_quant_method: str = "per-token",
        group_size: int = -1,
        **kwargs,
    ):
        super().__init__(
            wbits=wbits,
            w_quant_method=w_quant_method,
            a_quant_method=a_quant_method,
            group_size=group_size,
            **kwargs,
        )
        self.cfg = INT8DynamicConfig(
            wbits=int(wbits),
            w_quant_method=str(w_quant_method),
            a_quant_method=str(a_quant_method),
            group_size=int(group_size),
        )
        self._handlers: Dict[str, INT8LayerHandler] = {}

    def on_start(self) -> None:
        logger.info(
            f"[INT8Dynamic] start: wbits={self.cfg.wbits}, "
            f"method={self.cfg.w_quant_method}, group_size={self.cfg.group_size}"
        )
        self._handlers.clear()

    def on_finish(self) -> None:
        logger.info(f"[INT8Dynamic] finish: handlers={len(self._handlers)}")

    def _get_handler(self, key: str) -> INT8LayerHandler:
        handler = self._handlers.get(key)
        if handler is None:
            handler = INT8LayerHandler(
                wbits=self.cfg.wbits,
                method=self.cfg.w_quant_method,
                group_size=self.cfg.group_size,
            )
            self._handlers[key] = handler
        return handler

    def process_chunk(self, chunk: "ChunkContext") -> "ChunkContext":
        """
        Quantize all weight tensors in the chunk.

        Only processes floating-point weights (2D tensors ending with .weight).
        """
        for layer in chunk.layers:
            for tensor_name, tensor in list(layer.tensors.items()):
                # Only quantize floating-point weights
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                    or tensor.ndim < 2
                    or not tensor_name.endswith(".weight")
                ):
                    continue

                handler = self._get_handler(f"{layer.name}.{tensor_name}")
                layer.tensors[tensor_name] = handler.quantize(tensor.float()).to(tensor.dtype)

        return chunk
