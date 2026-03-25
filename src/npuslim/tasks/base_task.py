# src/npuslim/tasks/base_task.py
"""Base task contract and config registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from npuslim.algorithms.base_algo import AlgorithmConfig


# =============================================================================
# Task Config Registry
# =============================================================================

TASK_CONFIG_REGISTRY: Dict[str, Type["RecipeTaskConfig"]] = {}


def register_task_config(task_type: str, aliases: Optional[List[str]] = None):
    """Decorator to register a task-specific config class."""
    def decorator(cls: Type["RecipeTaskConfig"]) -> Type["RecipeTaskConfig"]:
        TASK_CONFIG_REGISTRY[task_type] = cls
        if aliases:
            for alias in aliases:
                TASK_CONFIG_REGISTRY[alias] = cls
        return cls
    return decorator


def get_task_config_class(task_type: str) -> Type["RecipeTaskConfig"]:
    """Get task config class by type, falling back to base class."""
    return TASK_CONFIG_REGISTRY.get(task_type, RecipeTaskConfig)


# =============================================================================
# Base Config
# =============================================================================

@dataclass
class RecipeTaskConfig:
    """Base recipe task configuration - common fields for all task types."""

    name: str
    type: str
    model: Optional[str] = None
    data: Optional[str] = None
    algorithm: Optional[AlgorithmConfig] = None
    saver: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Task
# =============================================================================

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
