"""Config parser for resources+recipe pattern."""
from typing import Any, Dict, Union
from pathlib import Path

import yaml

from npuslim.config.schema import (
    AlgorithmConfig,
    EngineConfig,
    MetadataConfig,
    RecipeTaskConfig,
    ResourceConfig,
    SlimConfig,
    TaskExecutionConfig,
)


def _normalize_execution_mode(mode: str) -> str:
    normalized = (mode or "full").lower()
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
                exec_cfg = TaskExecutionConfig(mode=_normalize_execution_mode(execution))
            elif isinstance(execution, dict):
                exec_cfg = TaskExecutionConfig(
                    mode=_normalize_execution_mode(execution.get("mode", "full")),
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
