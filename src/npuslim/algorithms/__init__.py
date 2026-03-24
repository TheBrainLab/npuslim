"""Algorithm framework for NPUSlim."""
from npuslim.algorithms.base import BaseAlgorithm, step, StepInfo
from npuslim.algorithms.registry import AlgorithmRegistry, register_algorithm

__all__ = [
    "BaseAlgorithm",
    "step",
    "StepInfo",
    "AlgorithmRegistry",
    "register_algorithm",
]
