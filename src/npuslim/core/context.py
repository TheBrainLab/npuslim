# src/npuslim/core/context.py
"""Algorithm context for NPUSlim."""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader

from npuslim.config.schema import Config, ExecutionMode, ChunkConfig

if TYPE_CHECKING:
    from npuslim.hooks import HookDispatcher
    from npuslim.streaming import StreamLoader, StreamSaver
    from npuslim.distributed import DistributedManager


class AlgorithmContext:
    """
    Layered context passed through algorithm execution.
    Provides access to streaming utilities, hooks, and intermediate storage.
    """

    def __init__(
        self,
        config: Config,
        model: Any,  # Model wrapper (BaseLLMModel)
        dataloader: Optional[DataLoader] = None,
        hooks: Optional["HookDispatcher"] = None,
    ):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.hooks = hooks

        # Streaming infrastructure (set by executor)
        self._stream_loader: Optional["StreamLoader"] = None
        self._stream_saver: Optional["StreamSaver"] = None

        # Distributed infrastructure (set by executor)
        self._distributed: Optional["DistributedManager"] = None

        # State management
        self._intermediates: Dict[str, Any] = {}
        self._current_chunk: Optional[Dict[str, Any]] = None
        self._layer_index: int = 0

    # === Properties ===

    @property
    def execution_mode(self) -> ExecutionMode:
        """Get current execution mode."""
        return self.config.execution_mode

    @property
    def is_streaming(self) -> bool:
        """Check if streaming mode is enabled."""
        return self.config.streaming is not None and self.config.streaming.enabled

    @property
    def is_distributed(self) -> bool:
        """Check if distributed mode is enabled."""
        return self._distributed is not None and self._distributed.is_distributed

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process in distributed execution."""
        if self._distributed is None:
            return True
        return self._distributed.is_main_process

    @property
    def chunk_config(self) -> Optional[ChunkConfig]:
        """Get chunk configuration."""
        return self.config.chunk

    @property
    def current_chunk(self) -> Optional[Dict[str, Any]]:
        """Get current chunk data."""
        return self._current_chunk

    @property
    def layer_index(self) -> int:
        """Get current layer index within chunk."""
        return self._layer_index

    @property
    def current_layer(self) -> Optional[Any]:
        """Get current layer object."""
        if self._current_chunk is None:
            return None
        layers = self._current_chunk.get("layers", [])
        if self._layer_index >= len(layers):
            return None
        return layers[self._layer_index]

    @property
    def current_layer_name(self) -> Optional[str]:
        """Get current layer name."""
        layer = self.current_layer
        if layer is None:
            return None
        # Handle both dict-style and attribute-style access
        if isinstance(layer, dict):
            return layer.get("name", "unknown")
        return getattr(layer, "name", "unknown")

    # === Layer Access ===

    def get_layers(self) -> List[Any]:
        """Get all layers to be quantized (in current chunk or full model)."""
        if self._current_chunk is not None:
            return self._current_chunk.get("layers", [])
        return self.model.get_layers()

    def get_total_layers(self) -> int:
        """Get total number of layers in model."""
        if hasattr(self.model, "get_total_layers"):
            return int(self.model.get_total_layers())
        return len(self.model.get_layers())

    def load_chunk(self, chunk_index: int, chunk_size: int) -> List[Any]:
        """Load one chunk from model runtime."""
        if hasattr(self.model, "load_chunk"):
            return self.model.load_chunk(chunk_index=chunk_index, chunk_size=chunk_size)

        # Fallback: slice from full layer list
        layers = self.model.get_layers()
        start = chunk_index * chunk_size
        end = min(start + chunk_size, len(layers))
        return layers[start:end]

    def release_chunk(self, chunk_index: int) -> None:
        """Release one chunk in model runtime."""
        if hasattr(self.model, "release_chunk"):
            self.model.release_chunk(chunk_index)

    # === Intermediate Storage ===

    def get_intermediate(self, key: str) -> Optional[Any]:
        """Get intermediate result from previous step."""
        return self._intermediates.get(key)

    def set_intermediate(self, key: str, value: Any) -> None:
        """Store intermediate result for next step."""
        self._intermediates[key] = value

    def clear_intermediates(self) -> None:
        """Clear all intermediates (called on chunk exit)."""
        self._intermediates.clear()

    # === Chunk Management ===

    def set_current_chunk(self, chunk: Dict[str, Any]) -> None:
        """Set the current chunk context."""
        self._current_chunk = chunk
        self._layer_index = 0

    def advance_layer(self) -> None:
        """Advance to next layer in current chunk."""
        self._layer_index += 1

    # === Streaming API ===

    def emit(self, name: str, tensor: torch.Tensor) -> None:
        """Emit a tensor to the streaming buffer (triggers flush if threshold exceeded)."""
        if self._stream_saver is None:
            raise RuntimeError("StreamSaver not initialized")
        self._stream_saver.add_tensor(name, tensor)

        if self.hooks:
            from npuslim.hooks import HookType
            self.hooks.dispatch(self, name=name, tensor=tensor)

    def flush(self) -> Optional[str]:
        """Manually flush the streaming buffer."""
        if self._stream_saver is None:
            return None
        return self._stream_saver.flush()
