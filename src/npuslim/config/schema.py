"""Configuration schema for NPUSlim."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionMode(Enum):
    """Execution mode for quantization."""

    FULL = "full"
    LAYER_WISE = "layer_wise"
    CHUNK_WISE = "chunk_wise"
    STREAMING = "streaming"


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
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    backend_init_method: str = "nccl"


@dataclass
class MetadataConfig:
    """Top-level metadata configuration."""

    name: str = ""
    description: str = ""


@dataclass
class ResourceConfig:
    """Resource declaration."""

    id: str
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    """Algorithm configuration within one recipe task."""

    type: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionConfig:
    """Task-level execution options."""

    mode: str = "full"
    chunk_size: int = 1


@dataclass
class RecipeTaskConfig:
    """One recipe task configuration."""

    name: str
    type: str
    model: Optional[str] = None
    data: Optional[str] = None
    main_model: Optional[str] = None
    draft_model: Optional[str] = None
    algorithm: Optional[AlgorithmConfig] = None
    execution: TaskExecutionConfig = field(default_factory=TaskExecutionConfig)
    saver: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    """Full NPUSlim YAML config."""

    metadata: MetadataConfig
    resources: List[ResourceConfig]
    recipe: List[RecipeTaskConfig]

    def get_resource_by_id(self, resource_id: str) -> Optional[ResourceConfig]:
        clean_id = resource_id.lstrip("@")
        for resource in self.resources:
            if resource.id == clean_id:
                return resource
        return None

    def get_resources_by_type(self, type_suffix: str) -> List[ResourceConfig]:
        return [resource for resource in self.resources if resource.type.endswith(type_suffix)]


SlimConfig = EngineConfig
