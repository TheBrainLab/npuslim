"""Base algorithm class with @step decorator."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from npuslim.core.context import AlgorithmContext


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

    def __init__(self, **kwargs):
        self.params: Dict[str, object] = dict(kwargs)
        self._steps: List[StepInfo] = []
        self._collect_steps()

    def _collect_steps(self) -> None:
        """Collect all @step decorated methods."""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_step_info"):
                info = method._step_info
                # Store bound method so StepExecutor can call it with (context, **inputs).
                self._steps.append(
                    StepInfo(
                        method=method,
                        order=info.order,
                        requires=list(info.requires),
                        produces=list(info.produces),
                    )
                )
        self._steps.sort(key=lambda s: s.order)

    def get_steps(self) -> List[StepInfo]:
        """Get all steps in execution order."""
        return self._steps

    def on_start(self, context: "AlgorithmContext") -> None:
        """Called when algorithm starts."""

    def on_chunk_enter(self, context: "AlgorithmContext") -> None:
        """Called when entering a new chunk."""

    def on_chunk_exit(self, context: "AlgorithmContext") -> None:
        """Called when exiting a chunk."""

    def on_finish(self, context: "AlgorithmContext") -> None:
        """Called when algorithm finishes."""
