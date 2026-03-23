"""
NPUSlim v2.0 - Streaming-first quantization framework.

This module provides a redesigned framework for memory-efficient
quantization of large language models.
"""

from npuslim.v2.config import (
    V2Config,
    ExecutionMode,
    ChunkConfig,
    StreamingConfig,
)
from npuslim.v2.hooks import (
    HookType,
    HookInfo,
    HookRegistry,
    HookDispatcher,
    register_hook,
)
from npuslim.v2.context import AlgorithmContext
from npuslim.v2.algorithm import BaseAlgorithm, step, StepInfo
from npuslim.v2.step_executor import StepExecutor
from npuslim.v2.streaming import StreamLoader, StreamSaver
from npuslim.v2.engine import SlimEngineV2, EngineConfig
from npuslim.v2.executor import PipelineExecutor

__all__ = [
    # Config
    "V2Config",
    "ExecutionMode",
    "ChunkConfig",
    "StreamingConfig",
    # Hooks
    "HookType",
    "HookInfo",
    "HookRegistry",
    "HookDispatcher",
    "register_hook",
    # Context
    "AlgorithmContext",
    # Algorithm
    "BaseAlgorithm",
    "step",
    "StepInfo",
    "StepExecutor",
    # Streaming
    "StreamLoader",
    "StreamSaver",
    # Engine
    "SlimEngineV2",
    "EngineConfig",
    "PipelineExecutor",
]
