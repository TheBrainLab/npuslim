"""Algorithm framework for NPUSlim."""

from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.algorithms.quantization import BaseQuantizationAlgorithm
from npuslim.registry import AlgorithmRegistry, register_algorithm

# Lazy registration for algorithms
AlgorithmRegistry.register_lazy(
    "INT8Dynamic",
    ".quantization.int8_dynamic.int8_dynamic_algo",
    aliases=["INT8Dyn", "int8_dyn"],
)
AlgorithmRegistry.register_lazy(
    "GPTQ",
    ".quantization.gptq.gptq_algo",
    aliases=["gptq", "GPTQStepwise", "GPTQExample", "gptq_stepwise"],
)

__all__ = [
    "BaseAlgorithm",
    "BaseQuantizationAlgorithm",
    "AlgorithmRegistry",
    "register_algorithm",
]
