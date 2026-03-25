# src/npuslim/__init__.py
"""NPUSlim - Streaming-first quantization framework."""

import importlib.metadata
from pathlib import Path

try:
    __version__ = importlib.metadata.version("npuslim")
except importlib.metadata.PackageNotFoundError:
    try:
        toml_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        if toml_path.exists():
            import re
            content = toml_path.read_text(encoding="utf-8")
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            __version__ = match.group(1) if match else "0.0.0-unknown"
        else:
            __version__ = "0.0.0-dev"
    except Exception:
        __version__ = "0.0.0-dev"

# Config schema
from npuslim.config import (
    AlgorithmConfig,
    DistributedBackend,
    DistributedConfig,
    EngineConfig,
    ExecutionConfig,
    MetadataConfig,
    RecipeTaskConfig,
    CompressorTaskConfig,
    ResourceConfig,
    ValidationError,
    parse_config,
    validate_config,
)

# Core runtime
from npuslim.core import (
    SlimEngine,
    ResourceManager,
)

# Algorithms
from npuslim.algorithms import BaseAlgorithm
from npuslim.registry import AlgorithmRegistry, register_algorithm

# Hooks
from npuslim.hooks import (
    HookType,
    HookInfo,
    HookRegistry,
    HookDispatcher,
    register_hook,
)

# Streaming
from npuslim.streaming import StreamLoader, StreamSaver

# Distributed
from npuslim.distributed import DistributedManager

# Registry
from npuslim.registry import (
    Registry,
    ModelRegistry,
    DatasetRegistry,
    TaskRegistry,
    SaverRegistry,
)

__all__ = [
    "__version__",
    # Config
    "MetadataConfig",
    "ResourceConfig",
    "AlgorithmConfig",
    "RecipeTaskConfig",
    "CompressorTaskConfig",
    "ExecutionConfig",
    "EngineConfig",
    "DistributedConfig",
    "DistributedBackend",
    "validate_config",
    "ValidationError",
    "parse_config",
    # Core
    "SlimEngine",
    "ResourceManager",
    # Algorithms
    "BaseAlgorithm",
    "AlgorithmRegistry",
    "register_algorithm",
    # Hooks
    "HookType",
    "HookInfo",
    "HookRegistry",
    "HookDispatcher",
    "register_hook",
    # Streaming
    "StreamLoader",
    "StreamSaver",
    # Distributed
    "DistributedManager",
    # Registry
    "Registry",
    "ModelRegistry",
    "DatasetRegistry",
    "TaskRegistry",
    "SaverRegistry",
]
