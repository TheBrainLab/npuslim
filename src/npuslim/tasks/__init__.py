"""Tasks for NPUSlim."""

from npuslim.tasks.base_task import BaseTask
from npuslim.registry import TaskRegistry

# Lazy registration for compressor task
TaskRegistry.register_lazy(
    "compressor",
    ".compressor.task",
    aliases=["CompressorTask", "QuantizeTask"],
)

__all__ = [
    "BaseTask",
    "TaskRegistry",
]
