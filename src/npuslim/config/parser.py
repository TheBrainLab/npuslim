"""Config parser for resources+recipe pattern."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import yaml


# Temporary stubs - will be replaced by imports from implementations
# TODO: Import from tasks.base_task, models.base_model, etc. when available


@dataclass
class MetadataConfig:
    """Metadata configuration."""
    name: str = ""
    description: str = ""


@dataclass
class ResourceConfig:
    """Resource configuration (raw, before type-specific parsing)."""
    id: str
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    """Algorithm configuration within a recipe task."""
    type: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionConfig:
    """Execution configuration for a recipe task."""
    mode: str = "full"
    chunk_size: int = 1


@dataclass
class RecipeTaskConfig:
    """Configuration for a single recipe task."""
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
    """Full NPUSlim configuration (resources + recipe pattern)."""
    metadata: MetadataConfig
    resources: List[ResourceConfig]
    recipe: List[RecipeTaskConfig]

    def get_resource_by_id(self, resource_id: str) -> Optional[ResourceConfig]:
        """Get a resource by its ID (with or without @ prefix)."""
        clean_id = resource_id.lstrip("@")
        for r in self.resources:
            if r.id == clean_id:
                return r
        return None

    def get_resources_by_type(self, type_suffix: str) -> List[ResourceConfig]:
        """Get all resources matching a type suffix (e.g., 'Model', 'Dataset')."""
        return [r for r in self.resources if r.type.endswith(type_suffix)]


def parse_config(source: Union[str, Path, Dict]) -> EngineConfig:
    """
    Parse YAML file or dict into EngineConfig.

    Args:
        source: Path to YAML file or dictionary

    Returns:
        EngineConfig instance
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = source

    return _parse_dict(data)


def _parse_dict(data: Dict) -> EngineConfig:
    """Parse dictionary into EngineConfig."""
    meta_data = data.get("metadata", {})
    metadata = MetadataConfig(
        name=meta_data.get("name", ""),
        description=meta_data.get("description", "")
    )

    resources = []
    for r in data.get("resources", []):
        r_copy = r.copy()
        resources.append(ResourceConfig(
            id=r_copy.pop("id"),
            type=r_copy.pop("type"),
            extra=r_copy
        ))

    recipe = []
    for t in data.get("recipe", []):
        t_copy = t.copy()
        algo = None
        if "algorithm" in t_copy:
            a = t_copy.pop("algorithm")
            if isinstance(a, str):
                algo = AlgorithmConfig(type=a)
            else:
                a_copy = a.copy()
                algo = AlgorithmConfig(type=a_copy.pop("type"), extra=a_copy)

        exec_cfg = TaskExecutionConfig()
        if "execution" in t_copy:
            execution = t_copy.pop("execution")
            if isinstance(execution, str):
                exec_cfg = TaskExecutionConfig(mode=execution)
            elif isinstance(execution, dict):
                exec_cfg = TaskExecutionConfig(
                    mode=execution.get("mode", "full"),
                    chunk_size=execution.get("chunk_size", 1),
                )

        recipe.append(RecipeTaskConfig(
            name=t_copy.pop("name"),
            type=t_copy.pop("type"),
            model=t_copy.pop("model", None),
            data=t_copy.pop("data", None),
            main_model=t_copy.pop("main_model", None),
            draft_model=t_copy.pop("draft_model", None),
            algorithm=algo,
            execution=exec_cfg,
            saver=t_copy.pop("saver", None),
            extra=t_copy
        ))

    return EngineConfig(metadata=metadata, resources=resources, recipe=recipe)


# Backward-compatible alias kept for config tests and downstream imports.
SlimConfig = EngineConfig
