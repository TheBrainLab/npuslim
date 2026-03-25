"""Factory registry for NPUSlim."""
from typing import Iterable, Optional

from npuslim.registry.factory import (
    Registry,
    AlgorithmRegistry,
    ModelRegistry,
    DatasetRegistry,
    TaskRegistry,
    SaverRegistry,
)


def register_algorithm(name: str, aliases: Optional[Iterable[str] | str] = None):
    """Decorator helper for algorithm registration."""
    return AlgorithmRegistry.register(name, aliases=aliases)


# Lazy registration for savers
SaverRegistry.register_lazy(
    "HuggingFaceSaver",
    "npuslim.savers.hf_saver",
    aliases=["hf", "HF"],
)

__all__ = [
    "Registry",
    "AlgorithmRegistry",
    "register_algorithm",
    "ModelRegistry",
    "DatasetRegistry",
    "TaskRegistry",
    "SaverRegistry",
]
