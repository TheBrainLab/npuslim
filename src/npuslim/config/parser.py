"""Config parser for resources+recipe pattern."""
from typing import Any, Dict, Union
from pathlib import Path

import yaml

from npuslim.config.schema import (
    AlgorithmConfig,
    CompressorTaskConfig,
    EngineConfig,
    ExecutionConfig,
    MetadataConfig,
    RecipeTaskConfig,
    ResourceConfig,
)


def _normalize_execution_mode(mode: str) -> str:
    normalized = (mode or "streaming").lower()
    if normalized == "stream":
        return "streaming"
    return normalized

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


def _parse_algorithm(a: Any) -> AlgorithmConfig:
    """Parse algorithm config."""
    if isinstance(a, str):
        return AlgorithmConfig(type=a)
    a_copy = a.copy()
    return AlgorithmConfig(type=a_copy.pop("type"), extra=a_copy)


def _parse_execution(execution: Any) -> ExecutionConfig:
    """Parse execution config for compressor tasks."""
    if isinstance(execution, str):
        return ExecutionConfig(mode=_normalize_execution_mode(execution))
    return ExecutionConfig(
        mode=_normalize_execution_mode(execution.get("mode", "streaming")),
        chunk_size=execution.get("chunk_size", 1),
    )


def _parse_task(t: Dict) -> RecipeTaskConfig:
    """Parse a single task config, returning appropriate task-specific config."""
    t_copy = t.copy()
    task_type = t_copy.pop("type", "")
    name = t_copy.pop("name", "")

    # Parse common fields
    model = t_copy.pop("model", None)
    data = t_copy.pop("data", None)
    algorithm = _parse_algorithm(t_copy.pop("algorithm")) if "algorithm" in t_copy else None
    saver = t_copy.pop("saver", None)

    # Create task-specific config based on type
    if task_type in ("compressor", "CompressorTask", "QuantizeTask"):
        # Compressor-specific fields
        ignore_layers = t_copy.pop("ignore_layers", [])
        execution = _parse_execution(t_copy.pop("execution", {})) if "execution" in t_copy else ExecutionConfig()

        return CompressorTaskConfig(
            name=name,
            type=task_type,
            model=model,
            data=data,
            algorithm=algorithm,
            saver=saver,
            ignore_layers=ignore_layers,
            execution=execution,
            extra=t_copy,  # remaining fields go to extra
        )
    else:
        # Default task config for unknown types
        return RecipeTaskConfig(
            name=name,
            type=task_type,
            model=model,
            data=data,
            algorithm=algorithm,
            saver=saver,
            extra=t_copy,
        )


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

    recipe = [_parse_task(t) for t in data.get("recipe", [])]

    return EngineConfig(metadata=metadata, resources=resources, recipe=recipe)
