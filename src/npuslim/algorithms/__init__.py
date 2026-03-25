"""Algorithm framework for NPUSlim."""

from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.registry import AlgorithmRegistry, register_algorithm

# Lazy registration for algorithms
AlgorithmRegistry.register_lazy(
    "INT8Dynamic",
    ".int8_dynamic.int8_dynamic_algo",
    aliases=["INT8Dyn", "int8_dyn"],
)

__all__ = [
    "BaseAlgorithm",
    "AlgorithmRegistry",
    "register_algorithm",
]
