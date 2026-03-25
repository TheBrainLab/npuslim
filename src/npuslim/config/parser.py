# src/npuslim/config/parser.py
"""Config parser for resources+recipe pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from npuslim.algorithms.base_algo import AlgorithmConfig
from npuslim.core.engine import EngineConfig
from npuslim.core.resource_config import MetadataConfig, ResourceConfig
from npuslim.tasks.base_task import (
    RecipeTaskConfig,
    get_task_config_class,
)


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
    a_copy = dict(a)
    algo_type = a_copy.pop("type")
    return AlgorithmConfig(type=algo_type, extra=a_copy)


def _parse_task(t: Dict) -> RecipeTaskConfig:
    """Parse a single task config, using registry to get task-specific config class."""
    t_copy = dict(t)
    task_type = t_copy.pop("type", "")
    task_name = t_copy.pop("name", "")

    # Common fields
    model = t_copy.pop("model", None)
    data = t_copy.pop("data", None)
    algorithm = _parse_algorithm(t_copy.pop("algorithm")) if "algorithm" in t_copy else None
    saver = t_copy.pop("saver", None)

    # Get task-specific config class from registry
    config_cls = get_task_config_class(task_type)

    # Task-specific parsing
    if task_type in ("compressor", "CompressorTask", "QuantizeTask"):
        from npuslim.tasks.compressor.task import ExecutionConfig

        ignore_layers = t_copy.pop("ignore_layers", [])
        execution_raw = t_copy.pop("execution", {})
        execution = ExecutionConfig(
            mode=execution_raw.get("mode", "streaming"),
            chunk_size=execution_raw.get("chunk_size", 1),
        )

        return config_cls(
            name=task_name,
            type=task_type,
            model=model,
            data=data,
            algorithm=algorithm,
            saver=saver,
            ignore_layers=ignore_layers,
            execution=execution,
            extra=t_copy,
        )

    # Default: use base config class
    return config_cls(
        name=task_name,
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
        description=meta_data.get("description", ""),
    )

    resources: List[ResourceConfig] = []
    for r in data.get("resources", []):
        r_copy = dict(r)
        res_id = r_copy.pop("id")
        res_type = r_copy.pop("type")
        resources.append(ResourceConfig(
            id=res_id,
            type=res_type,
            extra=r_copy,
        ))

    recipe: List[RecipeTaskConfig] = [
        _parse_task(t) for t in data.get("recipe", [])
    ]

    return EngineConfig(
        metadata=metadata,
        resources=resources,
        recipe=recipe,
    )
