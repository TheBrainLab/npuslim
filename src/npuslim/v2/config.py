# src/npuslim/v2/config.py
"""V2 configuration schema for NPUSlim."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionMode(Enum):
    """Execution mode for quantization."""
    FULL = "full"
    LAYER_WISE = "layer_wise"
    CHUNK_WISE = "chunk_wise"
    STREAMING = "streaming"


@dataclass
class ChunkConfig:
    """Configuration for chunk-based loading."""
    size: int = 1  # Number of transformer blocks per chunk
    offload_strategy: str = "lazy"  # "lazy" or "eager"
    preload_layers: Optional[int] = None  # Layers to preload into memory


@dataclass
class StreamingConfig:
    """Configuration for streaming output."""
    enabled: bool = True
    shard_size: str = "5GB"
    size_threshold: int = 4 * 1024 * 1024 * 1024  # 4 GiB
    output_dir: Optional[str] = None


@dataclass
class V2Config:
    """Main V2 configuration."""
    execution_mode: ExecutionMode = ExecutionMode.FULL
    chunk: Optional[ChunkConfig] = None
    streaming: Optional[StreamingConfig] = None

    def __post_init__(self):
        """Validate configuration compatibility."""
        if self.execution_mode == ExecutionMode.CHUNK_WISE and not self.chunk:
            raise ValueError("chunk config required for CHUNK_WISE mode")
        if self.execution_mode == ExecutionMode.LAYER_WISE and not self.streaming:
            # Warning only, not an error
            pass
