# src/npuslim/core/__init__.py
"""
NPUSlim Core Framework

This module provides the core runtime components for
memory-efficient processing of large language models.
"""

from npuslim.core.engine import SlimEngine
from npuslim.core.context import AlgorithmContext
from npuslim.core.step_executor import StepExecutor
from npuslim.core.resource_manager import ResourceManager
from npuslim.core.backend import BackendHandler, bh

__all__ = [
    "SlimEngine",
    "AlgorithmContext",
    "StepExecutor",
    "ResourceManager",
    "BackendHandler",
    "bh",
]
