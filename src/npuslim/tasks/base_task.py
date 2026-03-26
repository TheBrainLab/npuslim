# src/npuslim/tasks/base_task.py
"""Base task contract and config registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger
from torch.utils.data import DataLoader

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
        dataloader: Optional[Dict[str, Any]] = None,
        algorithm: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        resource_manager: Optional["ResourceManager"] = None,
        **kwargs,
    ):
        self.name = name
        self.model_ref = model
        self.dataloader_config = dataloader or {}
        if self.dataloader_config and not isinstance(self.dataloader_config, dict):
            raise ValueError("[Task] 'dataloader' must be a dict")
        self.data_ref = self.dataloader_config.get("dataset")
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

    def _create_data(self) -> None:
        """Create calibration dataloader from task `dataloader` config."""
        if not self.dataloader_config:
            self._calib_data = None
            return
        if self.rm is None:
            raise ValueError("resource_manager is required")
        if not self.data_ref:
            raise ValueError("[Task] dataloader.dataset is required")

        # VL models use processor, LLM models use tokenizer
        processor = getattr(self._model_obj, "processor", None)
        if processor is None:
            processor = getattr(self._model_obj, "tokenizer", None)
        dataset = self.rm.acquire_dataset(self.data_ref, processor=processor)

        loader_kwargs = {
            k: v for k, v in self.dataloader_config.items() if k != "dataset"
        }
        if "collate_fn" not in loader_kwargs and hasattr(dataset, "collate_fn"):
            loader_kwargs["collate_fn"] = getattr(dataset, "collate_fn")
        self._calib_data = DataLoader(dataset, **loader_kwargs)

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
