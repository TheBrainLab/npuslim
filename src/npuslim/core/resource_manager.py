"""Lazy resource manager for task-driven runtime."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from loguru import logger

from npuslim.core.resource_config import ResourceConfig
from npuslim.registry import DatasetRegistry, ModelRegistry


class ResourceManager:
    """Manage model/dataset resources lazily across tasks."""

    def __init__(self, resources: Iterable[ResourceConfig]):
        self._resource_configs: Dict[str, ResourceConfig] = {r.id: r for r in resources}
        self._instances: Dict[str, Any] = {}
        self._model_states: Dict[str, Dict[str, Any]] = {}

    def get_resource_config(self, ref: str) -> ResourceConfig:
        resource_id = ref.lstrip("@")
        if resource_id not in self._resource_configs:
            raise KeyError(f"Unknown resource reference: {ref}")
        return self._resource_configs[resource_id]

    def _resolve_resource_kind(self, type_name: str) -> str:
        """Resolve resource kind by registry lookup, not by naming convention."""
        is_model = True
        is_dataset = True

        try:
            ModelRegistry.get(type_name)
        except KeyError:
            is_model = False

        try:
            DatasetRegistry.get(type_name)
        except KeyError:
            is_dataset = False

        if is_model and is_dataset:
            raise TypeError(
                f"Ambiguous resource type '{type_name}': found in both ModelRegistry and DatasetRegistry"
            )
        if is_model:
            return "model"
        if is_dataset:
            return "dataset"
        raise TypeError(
            f"Unknown resource type '{type_name}': not registered as model or dataset"
        )

    def acquire_model(self, ref: str) -> Any:
        """Acquire model lazily by resource ref (e.g. @qwen3)."""
        resource_id = ref.lstrip("@")

        if resource_id in self._instances:
            return self._instances[resource_id]

        cfg = self.get_resource_config(ref)
        kind = self._resolve_resource_kind(cfg.type)
        if kind != "model":
            raise TypeError(f"Resource '{resource_id}' is not a model: {cfg.type}")

        model_kwargs = dict(cfg.extra)
        model = ModelRegistry.create(cfg.type, **model_kwargs)
        self._instances[resource_id] = model
        self._model_states[resource_id] = {
            "model": model,
            "source": "memory",
            "resource_id": resource_id,
        }
        logger.info(f"Lazy-loaded model resource: {resource_id} ({cfg.type})")
        return model

    def acquire_dataset(self, ref: str, processor: Any = None) -> Any:
        """Acquire dataset lazily by resource ref."""
        resource_id = ref.lstrip("@")

        if resource_id in self._instances:
            return self._instances[resource_id]

        cfg = self.get_resource_config(ref)
        kind = self._resolve_resource_kind(cfg.type)
        if kind != "dataset":
            raise TypeError(f"Resource '{resource_id}' is not a dataset: {cfg.type}")

        dataset_kwargs = dict(cfg.extra)
        if processor is not None:
            dataset_kwargs["processor"] = processor
        dataset = DatasetRegistry.create(cfg.type, **dataset_kwargs)

        self._instances[resource_id] = dataset
        logger.info(f"Lazy-loaded dataset resource: {resource_id} ({cfg.type})")
        return dataset

    def publish_model_state(self, ref: str, model_obj: Any, state_meta: Optional[Dict[str, Any]] = None) -> None:
        """Publish/update model state for downstream tasks."""
        resource_id = ref.lstrip("@")
        self._instances[resource_id] = model_obj
        payload = {
            "model": model_obj,
            "source": "memory",
            "resource_id": resource_id,
        }
        if state_meta:
            payload.update(state_meta)
        self._model_states[resource_id] = payload

    def get_model_state(self, ref: str) -> Optional[Dict[str, Any]]:
        """Get current tracked state for a model ref."""
        resource_id = ref.lstrip("@")
        return self._model_states.get(resource_id)

    def release_task_scope(self, task_name: str) -> None:
        """Task-scope cleanup hook (currently informational)."""
        logger.debug(f"Task scope released: {task_name}")
