"""
QuIP# (QuIP-Sharp) - Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.

Reference: https://arxiv.org/abs/2402.04396 (ICML 2024)
Source: https://github.com/Cornell-RelaxML/quip-sharp

Key features:
- Randomized Hadamard Transform (RHT) for incoherence processing
- E8 lattice codebooks for vector quantization (2-bit, 3-bit, 4-bit)
- Optional fine-tuning for improved fidelity
"""

from .quip_sharp import QuIPSharp, QuIPSharpConfig
from .quip_sharp_module import QuIPSharpModule
from .quip_sharp_linear import QuIPSharpLinear

__all__ = [
    "QuIPSharp",
    "QuIPSharpConfig",
    "QuIPSharpModule",
    "QuIPSharpLinear",
]
