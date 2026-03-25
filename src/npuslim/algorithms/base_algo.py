# src/npuslim/algorithms/base_algo.py
"""Base algorithm class and config."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from npuslim.tasks.compressor.context import ChunkContext



class BaseAlgorithm(ABC):
    """
    Base class for quantization algorithms.

    Algorithms implement process_chunk() which receives a ChunkContext
    containing layers with tensors, and returns the modified ChunkContext.
    """

    def __init__(self, **kwargs):
        self.params = dict(kwargs)

    @abstractmethod
    def process_chunk(self, chunk: "ChunkContext") -> "ChunkContext":
        """
        Process one chunk of layers.

        Args:
            chunk: Contains layers with tensors, calib_data, metadata

        Returns:
            Modified chunk (can modify in-place)
        """
        raise NotImplementedError

    def on_start(self) -> None:
        """Called before processing starts."""
        pass

    def on_finish(self) -> None:
        """Called after all chunks are processed."""
        pass
