"""INT8 Dynamic algorithm skeleton for v2 task/runtime flow debugging."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger

from npuslim.algorithms.base_algo import BaseAlgorithm, step
from npuslim.registry import register_algorithm


@dataclass
class INT8DynamicConfig:
    wbits: int = 8
    w_quant_method: str = "per-channel"
    a_quant_method: str = "per-token"


@register_algorithm("INT8Dynamic", aliases=["INT8Dyn", "int8_dyn"])
class INT8DynamicAlgorithm(BaseAlgorithm):
    """Payload-level INT8 dynamic conversion (no save/writeback)."""

    def __init__(
        self,
        wbits: int = 8,
        w_quant_method: str = "per-channel",
        a_quant_method: str = "per-token",
        **kwargs,
    ):
        super().__init__(
            wbits=wbits,
            w_quant_method=w_quant_method,
            a_quant_method=a_quant_method,
            **kwargs,
        )
        self.cfg = INT8DynamicConfig(
            wbits=int(wbits),
            w_quant_method=str(w_quant_method),
            a_quant_method=str(a_quant_method),
        )

    def _int_range(self) -> tuple[int, int]:
        bits = int(self.cfg.wbits)
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        return qmin, qmax

    def _quantize_dequantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            return tensor

        qmin, qmax = self._int_range()
        if self.cfg.w_quant_method == "per-channel" and tensor.ndim >= 2:
            # Linear-like tensors: scale over input dim per output channel.
            max_abs = tensor.abs().amax(dim=1, keepdim=True)
        else:
            max_abs = tensor.abs().amax()

        scale = max_abs / max(qmax, 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        dq = q * scale
        return dq.to(dtype=tensor.dtype)

    def _process_tensor_map(self, tensors: dict[str, torch.Tensor]) -> int:
        converted = 0
        for name, tensor in list(tensors.items()):
            if not name.endswith(".weight"):
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_floating_point():
                continue
            tensors[name] = self._quantize_dequantize_tensor(tensor)
            converted += 1
        return converted

    def _process_module_layer(self, layer_module) -> int:
        converted = 0
        with torch.no_grad():
            for _, submodule in layer_module.named_modules():
                if not isinstance(submodule, torch.nn.Linear):
                    continue
                weight = submodule.weight
                if weight is None or not weight.is_floating_point():
                    continue
                submodule.weight.copy_(self._quantize_dequantize_tensor(weight))
                converted += 1
        return converted

    def on_start(self, context):
        logger.info(
            f"[INT8Dynamic] start: mode={context.runtime.mode}, chunk_size={context.runtime.chunk_size}, "
            f"wbits={self.cfg.wbits}"
        )

    @step(order=0)
    def process_chunk(self, context):
        payload = context.current_chunk or {}
        layers = payload.get("layers", [])
        chunk_idx = payload.get("index", -1)
        converted = 0

        for layer in layers:
            if isinstance(layer, dict):
                tensors = layer.get("tensors", {})
                if isinstance(tensors, dict):
                    converted += self._process_tensor_map(tensors)
                continue
            converted += self._process_module_layer(layer)

        context.set_intermediate("converted_weights", converted)
        logger.info(
            f"[INT8Dynamic] chunk={chunk_idx}, layers={len(layers)}, converted_weights={converted}"
        )
        return {"converted_weights": converted}

    def on_finish(self, context):
        logger.info("[INT8Dynamic] finish")
