# src/npuslim/core/engine.py
"""SlimEngine main orchestrator."""

from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from npuslim.config import EngineConfig, parse_config
from npuslim.core.resource_manager import ResourceManager
from npuslim.registry import TaskRegistry


class SlimEngine:
    """
    Main orchestrator for NPUSlim.
    Manages global resources and pipeline execution.
    """

    def __init__(self, cfg_path: Union[str, Path]):
        self.cfg_path = Path(cfg_path)
        self.config: EngineConfig = parse_config(self.cfg_path)
        self.resources: Dict[str, Any] = {}
        self.pipeline: List[Any] = []
        self.resource_manager = ResourceManager(self.config.resources)

        logger.info(f"Loaded config from: {self.cfg_path}")
        logger.info(f"Recipe: {self.config.metadata.name}")

        # Initialize
        self._build_resources()
        self._build_pipeline()

    def _build_resources(self) -> None:
        """Index resources without eager initialization."""
        logger.info("Indexing resources for lazy loading...")
        self.resources = {r.id: r for r in self.config.resources}
        logger.info(f"  Indexed {len(self.resources)} resources")

    def _build_pipeline(self) -> None:
        """Build pipeline from config recipe."""
        logger.info("Building pipeline...")

        for task_config in self.config.recipe:
            try:
                task = self._create_task(task_config)
                self.pipeline.append(task)
                logger.info(f"  Added task: {task_config.name} ({task_config.type})")
            except NotImplementedError:
                logger.warning(f"  Task type '{task_config.type}' not implemented, skipping")
            except Exception as e:
                logger.error(f"  Failed to create task '{task_config.name}': {e}")

    def _create_task(self, task_config) -> Any:
        """Create a task from config."""
        task_kwargs = {**task_config.extra}
        task_kwargs["name"] = task_config.name
        task_kwargs["resource_manager"] = self.resource_manager

        # Pass resource references directly; tasks resolve lazily at runtime.
        for key in ["model", "data", "main_model", "draft_model"]:
            value = getattr(task_config, key, None)
            if value is not None:
                task_kwargs[key] = value

        execution = getattr(task_config, "execution", None)
        if execution is not None:
            task_kwargs["execution"] = execution

        # Add algorithm config if present
        if task_config.algorithm:
            task_kwargs["algorithm"] = task_config.algorithm

        # Add saver config if present
        if task_config.saver:
            task_kwargs["saver"] = task_config.saver

        return TaskRegistry.create(task_config.type, **task_kwargs)

    def run(self) -> None:
        """Execute the pipeline."""
        if not self.pipeline:
            logger.warning("Pipeline is empty. Nothing to run.")
            return

        logger.info(f"Starting pipeline with {len(self.pipeline)} tasks")

        for idx, task in enumerate(self.pipeline):
            task_name = task.__class__.__name__
            logger.info(f"Task {idx + 1}/{len(self.pipeline)}: {task_name}")
            task.execute()

        logger.success("Pipeline completed")
