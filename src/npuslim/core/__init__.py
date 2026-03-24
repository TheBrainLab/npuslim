# src/npuslim/core/__init__.py
"""
NPUSlim Core Framework

This module provides the core runtime components for
memory-efficient processing of large language models.
"""

from npuslim.core.engine import SlimEngine
from npuslim.core.executor import PipelineExecutor
from npuslim.core.context import AlgorithmContext
from npuslim.core.step_executor import StepExecutor

__all__ = [
    "SlimEngine",
    "PipelineExecutor",
    "AlgorithmContext",
    "StepExecutor",
]
