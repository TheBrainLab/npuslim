# src/npuslim/core/config.py
"""Core configuration schema for NPUSlim."""
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


class DistributedBackend(Enum):
    """Backend for distributed execution."""
    NONE = "none"
    ACCELERATE = "accelate"
    TORCH_DISTRIBUTED = "torch_distributed"
    DEEPSPEED = "deepspeed"


@dataclass
class DistributedConfig:
    """Configuration for distributed execution."""
    backend: DistributedBackend = DistributedBackend.NONE
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # Model parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Mixed precision (for accelerate)
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1

    # Communication
    backend_init_method: str = "nccl"  # "nccl", "gloo"


@dataclass
class Config:
    """Main configuration for NPUSlim framework."""
    execution_mode: ExecutionMode = ExecutionMode.FULL
    chunk: Optional[ChunkConfig] = None
    streaming: Optional[StreamingConfig] = None
    distributed: Optional[DistributedConfig] = None

    def __post_init__(self):
        """Validate configuration compatibility."""
        if self.execution_mode == ExecutionMode.CHUNK_WISE and not self.chunk:
            raise ValueError("chunk config required for CHUNK_WISE mode")
        if self.execution_mode == ExecutionMode.STREAMING and not self.streaming:
            raise ValueError("streaming config required for STREAMING mode")
