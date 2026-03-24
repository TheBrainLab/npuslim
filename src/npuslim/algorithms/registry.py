"""Algorithm registry."""
from typing import List, Optional

from npuslim.registry.factory import Registry


AlgorithmRegistry = Registry("Algorithm", "algorithms")


def register_algorithm(name: str, aliases: Optional[List[str]] = None):
    """Decorator helper for algorithm registration."""
    return AlgorithmRegistry.register(name, aliases=aliases)
