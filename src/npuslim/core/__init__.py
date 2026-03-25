# src/npuslim/core/__init__.py
"""
NPUSlim Core Framework

Minimal core for streaming quantization.
"""

from npuslim.core.backend import BackendHandler, bh
from npuslim.core.engine import SlimEngine
from npuslim.core.resource_manager import ResourceManager

__all__ = [
    "BackendHandler",
    "bh",
    "SlimEngine",
    "ResourceManager",
]
