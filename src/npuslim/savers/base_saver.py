# src/npuslim/savers/base_saver.py
"""Base saver interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class BaseSaver(ABC):
    """Base class for model savers."""

    @abstractmethod
    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add a tensor to be saved."""
        pass

    @abstractmethod
    def add_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Add multiple tensors."""
        pass

    @abstractmethod
    def flush(self) -> Optional[str]:
        """Flush current buffer to disk."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize saving, write index and metadata."""
        pass
