"""Base task contract for task-driven runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger


class BaseTask(ABC):
    """Base task with lifecycle hooks."""

    def __init__(self, *, name: str = "", **kwargs):
        self.name = name
        self.params: Dict[str, Any] = dict(kwargs)

    def on_start(self) -> None:
        """Called before task execution."""
        logger.info(f"[Task] Starting: {self.name or self.__class__.__name__}")

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
