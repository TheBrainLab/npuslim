# src/npuslim/tasks/base_task.py
"""Base task contract and config registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger

from npuslim.algorithms import BaseAlgorithm
from npuslim.registry import AlgorithmRegistry, SaverRegistry

if TYPE_CHECKING:
    from npuslim.core.resource_manager import ResourceManager
    from npuslim.savers.base_saver import BaseSaver


class BaseTask(ABC):
    """Base task with lifecycle hooks."""

    def __init__(
        self,
        *,
        name: str = "",
        model: Optional[str] = None,
        data: Optional[str] = None,
        algorithm: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        resource_manager: Optional["ResourceManager"] = None,
        **kwargs,
    ):
        self.name = name
        self.model_ref = model
        self.data_ref = data
        self.rm = resource_manager
        self.algorithm_config = algorithm or {}
        self.saver_config = saver or {}
        self.params = dict(kwargs)

        # Runtime state
        self._model_obj = None
        self._calib_data = None
        self._algorithm = None
        self._saver = None

    def _create_model(self):
        """Acquire model from resource manager."""
        if self.rm and self.model_ref:
            self._model_obj = self.rm.acquire_model(self.model_ref)

    def _create_data(self):
        """Acquire calibration data from resource manager."""
        if self.rm and self.data_ref:
            self._calib_data = self.rm.acquire_dataset(self.data_ref)

    def _create_algorithm(self) -> BaseAlgorithm:
        """Create algorithm from config."""
        algo_type = self.algorithm_config.get("type")
        if not algo_type:
            raise ValueError("Algorithm type not specified")
        algo_kwargs = {k: v for k, v in self.algorithm_config.items() if k != "type"}
        algo_cls = AlgorithmRegistry.get(algo_type)
        return algo_cls(**algo_kwargs)

    def _create_saver(self) -> Optional["BaseSaver"]:
        """Create saver from config."""
        if not self.saver_config:
            return None
        saver_type = self.saver_config.get("type", "StreamingHuggingFaceSaver")
        saver_kwargs = {k: v for k, v in self.saver_config.items() if k != "type"}
        return SaverRegistry.create(saver_type, **saver_kwargs)

    def on_start(self) -> None:
        """Called before task execution. Override to customize."""
        logger.info(f"[Task] Starting: {self.name or self.__class__.__name__}")
        self._create_model()
        self._create_data()
        self._algorithm = self._create_algorithm()
        self._saver = self._create_saver()

    def on_finish(self) -> None:
        """Called after task execution."""
        logger.info(f"[Task] Finished: {self.name or self.__class__.__name__}")

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Task-specific execution body."""
        raise NotImplementedError

    def execute(self) -> Dict[str, Any]:
        """Execute task with lifecycle hooks."""
        self.on_start()
        try:
            return self.run()
        finally:
            self.on_finish()
