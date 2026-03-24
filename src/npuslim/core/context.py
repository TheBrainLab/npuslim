# src/npuslim/core/context.py
"""Algorithm context (runtime state bag)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from npuslim.core.model_runtime import ModelRuntimeSession
    from npuslim.hooks import HookDispatcher
    from npuslim.streaming import StreamSaver


class AlgorithmContext:
    """Thin state carrier shared across algorithm execution."""

    def __init__(
        self,
        *,
        model: Any,
        dataloader: Optional[DataLoader] = None,
        runtime: "ModelRuntimeSession",
        saver: Optional["StreamSaver"] = None,
        hooks: Optional["HookDispatcher"] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.runtime = runtime
        self.hooks = hooks
        self._stream_saver = saver

        self._intermediates: Dict[str, Any] = {}
        self._payload: Optional[Dict[str, Any]] = None
        self._layer_index: int = 0

    @property
    def is_streaming(self) -> bool:
        return self.runtime.is_streaming and self._stream_saver is not None

    @property
    def current_chunk(self) -> Optional[Dict[str, Any]]:
        return self._payload

    @property
    def layer_index(self) -> int:
        return self._layer_index

    @property
    def current_layer(self) -> Optional[Any]:
        if self._payload is None:
            return None
        layers = self._payload.get("layers", [])
        if self._layer_index >= len(layers):
            return None
        return layers[self._layer_index]

    @property
    def current_layer_name(self) -> Optional[str]:
        layer = self.current_layer
        if layer is None:
            return None
        if isinstance(layer, dict):
            return layer.get("name", "unknown")
        return getattr(layer, "name", "unknown")

    def get_layers(self) -> List[Any]:
        if self._payload is None:
            return []
        return self._payload.get("layers", [])

    def get_total_layers(self) -> int:
        return int(self.runtime.get_total_layers())

    def load_chunk(self, chunk_index: int, chunk_size: int) -> List[Any]:
        return self.runtime.load_chunk(chunk_index=chunk_index, chunk_size=chunk_size)

    def release_chunk(self, chunk_index: int) -> None:
        self.runtime.release_chunk(chunk_index)

    def set_current_chunk(self, chunk: Dict[str, Any]) -> None:
        self._payload = chunk
        self._layer_index = 0

    def clear_current_chunk(self) -> None:
        self._payload = None
        self._layer_index = 0

    def set_layer_index(self, idx: int) -> None:
        self._layer_index = max(int(idx), 0)

    def advance_layer(self) -> None:
        self._layer_index += 1

    def get_intermediate(self, key: str) -> Optional[Any]:
        return self._intermediates.get(key)

    def set_intermediate(self, key: str, value: Any) -> None:
        self._intermediates[key] = value

    def clear_intermediates(self) -> None:
        self._intermediates.clear()

    def emit(self, name: str, tensor: torch.Tensor) -> None:
        if self._stream_saver is None:
            raise RuntimeError("StreamSaver not initialized")
        self._stream_saver.add_tensor(name, tensor)

    def flush(self) -> Optional[str]:
        if self._stream_saver is None:
            return None
        return self._stream_saver.flush()
