# src/npuslim/config/parser.py
"""Config parser for resources+recipe pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from npuslim.config.schema import (
    EngineConfig,
    MetadataConfig,
    RecipeTaskConfig,
    ResourceConfig,
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



def _parse_task(t: Dict) -> RecipeTaskConfig:
    """Parse a single task config, using registry to get task-specific config class."""
    t_copy = dict(t)
    task_type = t_copy.pop("type", "")
    task_name = t_copy.pop("name", "")

    # Parse common fields
    model = t_copy.pop("model", None)
    dataloader = t_copy.pop("dataloader", None)
    algorithm = t_copy.pop("algorithm", None)
    saver = t_copy.pop("saver", None)

    return RecipeTaskConfig(
        name=task_name,
        type=task_type,
        model=model,
        dataloader=dataloader,
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
        resources.append(
            ResourceConfig(
                id=res_id,
                type=res_type,
                extra=r_copy,
            )
        )

    recipe: List[RecipeTaskConfig] = [_parse_task(t) for t in data.get("recipe", [])]

    return EngineConfig(
        metadata=metadata,
        resources=resources,
        recipe=recipe,
    )
