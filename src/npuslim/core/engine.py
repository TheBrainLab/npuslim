# src/npuslim/core/engine.py
"""Simple pipeline orchestrator.

Responsibilities:
1. Parse YAML config into runtime objects.
2. Create ResourceManager from config resources.
3. Execute tasks in order, passing resource_manager via kwargs.
"""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from npuslim.config import parse_config
from npuslim.config.schema import AlgorithmConfig, ExecutionConfig
from npuslim.core.resource_manager import ResourceManager
from npuslim.registry import TaskRegistry


def _flatten_value(value: Any) -> Any:
    """Flatten nested config dataclasses, merging 'extra' into parent."""
    if isinstance(value, AlgorithmConfig):
        # Flatten AlgorithmConfig: {type, extra} -> {type, **extra}
        result = {"type": value.type}
        if value.extra:
            result.update(value.extra)
        return result
    elif isinstance(value, ExecutionConfig):
        # Convert to dict
        return asdict(value)
    elif isinstance(value, dict):
        return {k: _flatten_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_flatten_value(item) for item in value]
    return value


def _task_config_to_kwargs(config: Any) -> Dict[str, Any]:
    """Convert task config dataclass to kwargs dict."""
    result = {}
    for f in fields(config):
        if f.name in ("name", "type"):
            continue
        value = getattr(config, f.name)
        if value is not None:
            result[f.name] = _flatten_value(value)

    # Merge extra fields at the end (they can override)
    if config.extra:
        result.update(config.extra)

    return result


class SlimEngine:
    """
    Simple task runner - loads config, creates tasks, executes them.

    Tasks receive `resource_manager` via kwargs and decide what resources
    to acquire themselves. This keeps the engine minimal and tasks flexible.
    """

    def __init__(self, cfg_path: Union[str, Path]):
        self.cfg_path = Path(cfg_path)
        self.config = parse_config(self.cfg_path)
        self.pipeline: List[Any] = []

        # Create resource manager from config resources
        self.rm = ResourceManager(resources=getattr(self.config, "resources", []))

        logger.info(f"Loaded config from: {self.cfg_path}")
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build pipeline from config recipe."""
        for task_config in self.config.recipe:
            try:
                # Convert task config to kwargs
                task_kwargs = _task_config_to_kwargs(task_config)

                # Add resource_manager
                task_kwargs["resource_manager"] = self.rm

                task = TaskRegistry.create(
                    task_config.type,
                    name=task_config.name,
                    **task_kwargs,
                )
                self.pipeline.append(task)
                logger.info(f"  Added task: {task_config.name} ({task_config.type})")
            except Exception as e:
                logger.error(f"  Failed to create task '{task_config.name}': {e}")
                raise

    def run(self) -> List[Dict[str, Any]]:
        """Execute the pipeline."""
        if not self.pipeline:
            logger.warning("Pipeline is empty. Nothing to run.")
            return []

        logger.info(f"Starting pipeline with {len(self.pipeline)} tasks")
        results = []

        for idx, task in enumerate(self.pipeline):
            task_name = getattr(task, "name", None) or task.__class__.__name__
            logger.info(f"Task {idx + 1}/{len(self.pipeline)}: {task_name}")
            result = task.execute()
            results.append(result)

        logger.success("Pipeline completed")
        return results
