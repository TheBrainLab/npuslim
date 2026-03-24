# src/npuslim/core/engine.py
"""SlimEngine main orchestrator."""

from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from npuslim.config import EngineConfig, parse_config
from npuslim.registry import ModelRegistry, DatasetRegistry, TaskRegistry


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

        logger.info(f"Loaded config from: {self.cfg_path}")
        logger.info(f"Recipe: {self.config.metadata.name}")

        # Initialize
        self._build_resources()
        self._build_pipeline()

    def _build_resources(self) -> None:
        """Build resources from config."""
        logger.info("Building resources...")

        # Group resources by type
        models = self.config.get_resources_by_type("Model")
        datasets = self.config.get_resources_by_type("Dataset")

        # Initialize models
        for model_cfg in models:
            try:
                model = ModelRegistry.create(
                    model_cfg.type,
                    **model_cfg.extra
                )
                self.resources[model_cfg.id] = model
                logger.info(f"  Created model: {model_cfg.id} ({model_cfg.type})")
            except KeyError as e:
                logger.warning(f"  Model type '{model_cfg.type}' not registered, skipping: {e}")

        # Initialize datasets
        for dataset_cfg in datasets:
            try:
                dataset = DatasetRegistry.create(
                    dataset_cfg.type,
                    **dataset_cfg.extra
                )
                self.resources[dataset_cfg.id] = dataset
                logger.info(f"  Created dataset: {dataset_cfg.id} ({dataset_cfg.type})")
            except KeyError as e:
                logger.warning(f"  Dataset type '{dataset_cfg.type}' not registered, skipping: {e}")

        # Extract tokenizer from first model if available
        first_model = next(iter(self.resources.values()), None)
        if first_model and hasattr(first_model, "tokenizer"):
            self.resources["tokenizer"] = first_model.tokenizer
        elif first_model and hasattr(first_model, "processor"):
            self.resources["tokenizer"] = first_model.processor

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
        # Resolve resource references (e.g., "@qwen3" -> actual resource)
        resolved_resources = {}
        for key in ["model", "data", "main_model", "draft_model"]:
            ref = getattr(task_config, key, None)
            if ref and ref.startswith("@"):
                resource_id = ref[1:]
                if resource_id in self.resources:
                    resolved_resources[key] = self.resources[resource_id]

        # Merge resolved resources with extra config
        task_kwargs = {**task_config.extra, **resolved_resources}

        # Apply per-task execution mode to referenced models when supported.
        execution = getattr(task_config, "execution", None)
        if execution is not None:
            task_kwargs["execution"] = execution
            for model_key in ["model", "main_model", "draft_model"]:
                model_obj = resolved_resources.get(model_key)
                if model_obj is None or not hasattr(model_obj, "configure_runtime"):
                    continue
                model_obj.configure_runtime(
                    mode=execution.mode,
                    chunk_size=execution.chunk_size,
                )

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

