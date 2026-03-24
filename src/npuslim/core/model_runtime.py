"""Task-scoped model runtime session."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from npuslim.core.backend import bh
from npuslim.streaming import SafeTensorStreamLoader


class ModelRuntimeSession:
    """Owns model runtime mode, chunk policy, and loader lifecycle for one task."""

    def __init__(self, model: Any, mode: str = "full", chunk_size: int = 1):
        self.model = model
        self.mode = "full"
        self.chunk_size = max(int(chunk_size), 1)
        self.tensor_device = self._resolve_tensor_device()
        logger.info(
            f"[ModelRuntimeSession] tensor_device resolved from model_kwargs.device_map -> {self.tensor_device}"
        )
        self._stream_loader = SafeTensorStreamLoader(
            model_path=model.path_str,
            model_hub=model.model_hub,
            model_kwargs=model.model_kwargs,
            tokenizer_kwargs=model.tokenizer_kwargs,
            block_name=model.block_name,
            tensor_device=self.tensor_device,
        )
        self._refresh_loader_index()
        self.configure(mode=mode, chunk_size=self.chunk_size)

    def _resolve_tensor_device(self) -> str:
        """
        Resolve chunk tensor loading device from model_kwargs.device_map.
        Supports cpu/cuda/npu-style values and simple device-map dicts.
        """
        device_map = getattr(self.model, "model_kwargs", {}).get("device_map")
        return bh.resolve_device_map(device_map, default="cpu")

    def _total_layers_hint(self) -> Optional[int]:
        getter = getattr(self.model, "get_total_layers_from_config", None)
        if callable(getter):
            return getter()
        return None

    def _refresh_loader_index(self) -> None:
        self._stream_loader.set_block_name(self.model.block_name)
        self._stream_loader.refresh_index(total_layers_hint=self._total_layers_hint())

    def _resolve_auto_mode(self) -> str:
        budget = getattr(self.model, "auto_memory_budget_bytes", None)
        if budget is None:
            return "full"
        return "streaming" if int(self._stream_loader.total_size) > int(budget) else "full"

    def configure(self, mode: str = "full", chunk_size: Optional[int] = None) -> None:
        if chunk_size is not None:
            self.chunk_size = max(int(chunk_size), 1)

        normalized = (mode or "full").lower()
        if normalized not in {"full", "streaming", "auto"}:
            raise ValueError(f"Unsupported runtime mode: {mode}")

        self.mode = self._resolve_auto_mode() if normalized == "auto" else normalized
        if self.mode == "full":
            self.model.prepare_full_model(pretrained_source=self._stream_loader.resolve_model_source())
        else:
            self.model.release_full_model()

    @property
    def is_streaming(self) -> bool:
        return self.mode == "streaming"

    def get_total_layers(self) -> int:
        if self.mode == "full" and self.model.model is not None:
            return len(self.model.get_layers())
        return self._stream_loader.get_total_layers(total_layers_hint=self._total_layers_hint())

    def get_chunk_count(self, chunk_size: Optional[int] = None) -> int:
        size = max(int(chunk_size or self.chunk_size), 1)
        return self._stream_loader.get_chunk_count(
            chunk_size=size,
            total_layers_hint=self._total_layers_hint(),
        )

    def load_chunk(self, chunk_index: int, chunk_size: Optional[int] = None):
        size = max(int(chunk_size or self.chunk_size), 1)
        if self.mode == "full":
            self.model.prepare_full_model(pretrained_source=self._stream_loader.resolve_model_source())
            layers = self.model.get_layers()
            start = chunk_index * size
            end = min(start + size, len(layers))
            return layers[start:end]
        return self._stream_loader.load_chunk(chunk_index=chunk_index, chunk_size=size)

    def release_chunk(self, chunk_index: int) -> None:
        if self.mode == "streaming":
            self._stream_loader.unload_chunk(chunk_index)

    def close(self) -> None:
        self._stream_loader.close()
