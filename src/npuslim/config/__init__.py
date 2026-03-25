# src/npuslim/config/__init__.py
"""NPUSlim Config Module - re-exports from co-located modules."""

# Parser
from npuslim.config.parser import parse_config

# Engine configs
from npuslim.core.engine import EngineConfig
from npuslim.core.resource_config import MetadataConfig, ResourceConfig

# Task configs
from npuslim.tasks.base_task import RecipeTaskConfig, register_task_config
from npuslim.tasks.compressor.task import CompressorTaskConfig, ExecutionConfig

# Algorithm config
from npuslim.algorithms.base_algo import AlgorithmConfig

__all__ = [
    # Parser
    "parse_config",
    # Engine configs
    "EngineConfig",
    "MetadataConfig",
    "ResourceConfig",
    # Task configs
    "RecipeTaskConfig",
    "CompressorTaskConfig",
    "ExecutionConfig",
    "register_task_config",
    # Algorithm config
    "AlgorithmConfig",
]
