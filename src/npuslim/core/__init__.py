# src/npuslim/core/__init__.py
"""
NPUSlim Core Framework

Minimal core for streaming quantization.
"""

from npuslim.core.backend import BackendHandler, bh
from npuslim.core.bootstrap import bootstrap_from_path
from npuslim.core.engine import SlimEngine
from npuslim.core.factory import (
    AlgorithmRegistry,
    DatasetRegistry,
    ModelRegistry,
    Registry,
    SaverRegistry,
    TaskRegistry,
)
from npuslim.core.resource_manager import ResourceManager

__all__ = [
    "BackendHandler",
    "bh",
    "bootstrap_from_path",
    "SlimEngine",
    "ResourceManager",
    "Registry",
    "AlgorithmRegistry",
    "ModelRegistry",
    "DatasetRegistry",
    "TaskRegistry",
    "SaverRegistry",
]
