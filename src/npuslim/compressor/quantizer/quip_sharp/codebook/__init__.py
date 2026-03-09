"""
E8 Lattice Codebooks for QuIP# vector quantization.

Codebook types:
- E8P12: 2-bit quantization using E8 lattice
- E8P12RVQ3B: 3-bit using residual vector quantization (TODO)
- E8P12RVQ4B: 4-bit using residual vector quantization
"""

from .base_codebook import BaseCodebook
from .e8p12 import E8P12Codebook
from .e8p12_rvq4bit import E8P12RVQ4BCodebook

# Registry mapping codebook names to classes
_CODEBOOK_REGISTRY = {
    "E8P12": E8P12Codebook,
    "E8P12RVQ4B": E8P12RVQ4BCodebook,
    # "E8P12RVQ3B": E8P12RVQ3BCodebook,  # TODO: implement
}


def get_codebook(name: str, inference: bool = False):
    """
    Get codebook by name.

    Args:
        name: Codebook name (e.g., "E8P12", "E8P12RVQ4B")
        inference: If True, create inference-only codebook (smaller memory)

    Returns:
        Codebook instance
    """
    if name not in _CODEBOOK_REGISTRY:
        raise ValueError(f"Unknown codebook: {name}. Available: {list(_CODEBOOK_REGISTRY.keys())}")

    return _CODEBOOK_REGISTRY[name](inference=inference)


def get_codebook_id(name: str) -> int:
    """Get integer ID for codebook name."""
    ids = {name: i for i, name in enumerate(_CODEBOOK_REGISTRY.keys())}
    return ids.get(name, 0)


__all__ = [
    "BaseCodebook",
    "E8P12Codebook",
    "E8P12RVQ4BCodebook",
    "get_codebook",
    "get_codebook_id",
]
