# src/npuslim/config/schema.py
"""Configuration schema for NPUSlim.

Pure data structures - NO heavy imports (torch, safetensors, etc.).
This ensures fast config parsing without loading ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Metadata & Resources
# =============================================================================

@dataclass
class MetadataConfig:
    """Top-level metadata configuration."""

    name: str = ""
    description: str = ""


@dataclass
class ResourceConfig:
    """Resource declaration (model, dataset, etc.)."""

    id: str
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Task Config
# =============================================================================

@dataclass
class RecipeTaskConfig:
    """Base recipe task configuration - common fields for all task types.

    Subclasses should call super().__init__() and pass **kwargs to capture
    unknown fields into the `extra` dict.
    """

    name: str
    type: str
    model: Optional[str] = None
    dataloader: Optional[Dict[str, Any]] = None
    algorithm: Optional[Dict[str, Any]] = None
    saver: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Distributed Config
# =============================================================================

class DistributedBackend(Enum):
    """Backend for distributed execution."""

    NONE = "none"
    ACCELERATE = "accelerate"
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


# =============================================================================
# Engine Config
# =============================================================================

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
