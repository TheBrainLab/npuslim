# src/npuslim/v2/algorithm.py
"""Base algorithm class with @step decorator."""
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from npuslim.v2.config import V2Config, ExecutionMode

if TYPE_CHECKING:
    from npuslim.v2.context import AlgorithmContext


@dataclass
class StepInfo:
    """Information about a step method."""
    method: Callable
    order: int
    requires: List[str]
    produces: List[str]


def step(
    order: int,
    requires: Optional[List[str]] = None,
    produces: Optional[List[str]] = None,
):
    """Decorator to mark a method as a quantization step."""
    def decorator(func: Callable) -> Callable:
        step_info = StepInfo(
            method=func,
            order=order,
            requires=requires or [],
            produces=produces or [],
        )
        func._step_info = step_info
        return func
    return decorator


class BaseAlgorithm(ABC):
    """
    Base class for all quantization algorithms.
    Algorithms declare steps via @step decorator.
    Framework auto-executes steps in order.
    """

    # Subclasses should override these
    execution_mode: ExecutionMode = ExecutionMode.FULL
    chunk_size: int = 1

    def __init__(self, config: V2Config):
        self.config = config
        self._steps: List[StepInfo] = []
        self._collect_steps()

    def _collect_steps(self) -> None:
        """Collect all @step decorated methods."""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_step_info"):
                self._steps.append(method._step_info)
        # Sort by order
        self._steps.sort(key=lambda s: s.order)

    def get_steps(self) -> List[StepInfo]:
        """Get all steps in execution order."""
        return self._steps

    # Lifecycle hooks (subclasses can override)
    def on_start(self, context: "AlgorithmContext") -> None:
        """Called when algorithm starts."""
        pass

    def on_chunk_enter(self, context: "AlgorithmContext") -> None:
        """Called when entering a new chunk."""
        pass

    def on_chunk_exit(self, context: "AlgorithmContext") -> None:
        """Called when exiting a chunk."""
        pass

    def on_finish(self, context: "AlgorithmContext") -> None:
        """Called when algorithm finishes."""
        pass
