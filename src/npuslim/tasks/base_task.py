"""Base task contract for task-driven runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from npuslim.algorithms import AlgorithmRegistry, BaseAlgorithm


class BaseTask(ABC):
    """Base task with lazy resource acquisition helpers."""

    def __init__(
        self,
        *,
        name: str = "",
        resource_manager,
        execution: Optional[Any] = None,
        model: Any = None,
        data: Any = None,
        algorithm: Optional[Any] = None,
        saver: Optional[Any] = None,
        **kwargs,
    ):
        self.name = name
        self.resource_manager = resource_manager
        self.execution = execution
        self.model_ref = model
        self.data_ref = data
        self.algorithm = algorithm
        self.saver = saver
        self.extra = kwargs

    def _resolve_model(self, model_ref: Any) -> Any:
        if model_ref is None:
            return None
        if isinstance(model_ref, str) and model_ref.startswith("@"):
            return self.resource_manager.acquire_model(model_ref)
        return model_ref

    def _resolve_dataset(self, data_ref: Any, processor: Any = None) -> Any:
        if data_ref is None:
            return None
        if isinstance(data_ref, str) and data_ref.startswith("@"):
            return self.resource_manager.acquire_dataset(data_ref, processor=processor)
        return data_ref

    def _resolve_algorithm(self) -> Any:
        if self.algorithm is None:
            return None
        if isinstance(self.algorithm, BaseAlgorithm):
            return self.algorithm

        algorithm_type = getattr(self.algorithm, "type", None)
        if not algorithm_type:
            return None

        try:
            algorithm_cls = AlgorithmRegistry.get(algorithm_type)
        except KeyError:
            raise ValueError(f"Algorithm '{algorithm_type}' is not registered.")

        algorithm_kwargs = dict(getattr(self.algorithm, "extra", {}) or {})
        return algorithm_cls(**algorithm_kwargs)

    def on_start(self) -> None:
        logger.info(f"Task start: {self.name}")

    def on_finish(self) -> None:
        logger.info(f"Task finish: {self.name}")
        self.resource_manager.release_task_scope(self.name)

    @abstractmethod
    def run(self) -> Any:
        """Task-specific execution body."""

    def execute(self) -> Any:
        self.on_start()
        try:
            return self.run()
        finally:
            self.on_finish()
