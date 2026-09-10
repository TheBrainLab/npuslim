"""Algorithm framework for NPUSlim."""

from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.algorithms.quantization import BaseQuantizationAlgorithm
from npuslim.core import AlgorithmRegistry

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
AlgorithmRegistry.register_lazy(
    "SparseGPT",
    ".quantization.sparsegpt.sparsegpt_algo",
    aliases=["sparsegpt", "sparse_gpt"],
)
AlgorithmRegistry.register_lazy(
    "QuIP",
    ".quantization.quip.quip_algo",
    aliases=["quip"],
)
# A7 Step B：GPTQ + R1 旋转（独立子类，默认关闭，不影响既有 GPTQ 产线）
AlgorithmRegistry.register_lazy(
    "GPTQRotation",
    "npuslim.qad.rotation_quant",
    aliases=["gptq_rotation", "gptq_rot"],
)

__all__ = [
    "BaseAlgorithm",
    "BaseQuantizationAlgorithm",
    "AlgorithmRegistry",
]
