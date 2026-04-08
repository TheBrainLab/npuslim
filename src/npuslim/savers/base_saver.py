# src/npuslim/savers/base_saver.py
"""Base saver interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch


class BaseSaver(ABC):
    """Base class for model savers."""

    def __init__(
        self,
        *,
        save_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        save_dir: Path | str | None = None,
    ) -> None:
        self.output_dir = self.resolve_output_dir(
            save_path=save_path,
            output_dir=output_dir,
            save_dir=save_dir,
            saver_name=self.__class__.__name__,
        )

    @staticmethod
    def resolve_output_dir(
        *,
        save_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        save_dir: Path | str | None = None,
        saver_name: str = "BaseSaver",
    ) -> Path:
        """Resolve output directory with precedence: save_path > output_dir > save_dir."""
        resolved = save_path if save_path is not None else output_dir
        if resolved is None:
            resolved = save_dir
        if resolved is None:
            raise ValueError(
                f"{saver_name} requires 'save_path' or 'output_dir' "
                "(or legacy 'save_dir')."
            )
        return Path(resolved)

    @abstractmethod
    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        tensor_type: Optional[str] = None,
    ) -> None:
        """Add a tensor to be saved."""
        pass

    @abstractmethod
    def add_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        tensor_types: Optional[Dict[str, str]] = None,
    ) -> None:
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
