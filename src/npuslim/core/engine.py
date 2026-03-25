# src/npuslim/core/engine.py
"""Simple pipeline orchestrator.

Responsibilities:
1. Parse YAML config into runtime objects.
2. Create ResourceManager from config resources.
3. Execute tasks in order, passing resource_manager via kwargs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from npuslim.config.schema import EngineConfig, ResourceConfig
from npuslim.core.resource_manager import ResourceManager
from npuslim.registry import TaskRegistry


class SlimEngine:
    """
    Simple task runner - loads config, creates tasks, executes them.

    Tasks receive `resource_manager` via kwargs and decide what resources
    to acquire themselves. This keeps the engine minimal and tasks flexible.
    """

    def __init__(self, config: Union[str, Path, EngineConfig]):
        if isinstance(config, (str, Path)):
            from npuslim.config.parser import parse_config
            self.config = parse_config(config)
            self.cfg_path = Path(config)
        else:
            self.config = config
            self.cfg_path = None

        self.pipeline: List[Any] = []
        self.rm = ResourceManager(resources=self.config.resources)

        if self.cfg_path:
            logger.info(f"Loaded config from: {self.cfg_path}")
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build pipeline from config recipe."""
        for task_config in self.config.recipe:
            try:
                # Get kwargs from config (config handles its own flattening)
                task_kwargs = task_config.extra
                task = TaskRegistry.create(
                    type_name=task_config.type,
                    name=task_config.name,
                    model=task_config.model,
                    data=task_config.data,
                    algorithm=task_config.algorithm,
                    saver=task_config.saver,
                    resource_manager=self.rm,
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
