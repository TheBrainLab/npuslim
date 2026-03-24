"""Algorithm framework for NPUSlim."""
from npuslim.algorithms.base_algo import BaseAlgorithm, step, StepInfo
from npuslim.registry import AlgorithmRegistry, register_algorithm

AlgorithmRegistry.register_lazy(
    "INT8Dynamic",
    ".int8_dynamic",
    aliases=["INT8Dyn", "int8_dyn"],
)

__all__ = [
    "BaseAlgorithm",
    "step",
    "StepInfo",
    "AlgorithmRegistry",
    "register_algorithm",
]
