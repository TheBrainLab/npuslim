"""
Base class for QuIP# codebooks.

Codebooks implement vector quantization using lattice-based codebooks.
The E8 lattice provides optimal 8-dimensional ball packing.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseCodebook(nn.Module, ABC):
    """
    Abstract base class for QuIP# codebooks.

    Codebooks map continuous vectors to discrete lattice points.
    Key properties:
    - codesz: Dimension of each code vector (8 for E8 lattice)
    - idx_dtype: Data type for storing codebook indices
    - packs: Number of codes packed together for efficient storage
    """

    # Subclasses must define these
    codesz: int = 8  # Code size (E8 lattice is 8-dimensional)
    idx_dtype: torch.dtype = torch.int64
    packs: int = 4  # Pack factor for compressed storage
    pack_out: bool = False
    version: int = 1

    # Optimal scale for the codebook (determined empirically)
    opt_scale: float = 1.0

    def __init__(self, inference: bool = False):
        """
        Initialize codebook.

        Args:
            inference: If True, only load inference-needed buffers (saves memory)
        """
        super().__init__()
        self.inference = inference

    @abstractmethod
    def quantize(
        self,
        X: torch.Tensor,
        return_idx: bool = True,
        resid_scale_override: float = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize vectors to codebook entries.

        Args:
            X: Input tensor [*, codesz] - last dimension must be divisible by codesz
            return_idx: If True, return codebook indices
            resid_scale_override: Override residual scale (-1 for default)

        Returns:
            quantized: Quantized tensor [*, codesz]
            idx: Codebook indices [*, n_codes] (if return_idx=True)
        """
        pass

    @abstractmethod
    def by_idxs(self, idxs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decompress codebook indices to vectors.

        Args:
            idxs: Codebook indices

        Returns:
            Decompressed tensor
        """
        pass

    def maybe_pack_idxs(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Pack indices for efficient storage (optional).

        Args:
            idxs: Unpacked indices

        Returns:
            Packed indices
        """
        return idxs

    def maybe_unpack_idxs(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Unpack indices (optional, inverse of maybe_pack_idxs).

        Args:
            idxs: Packed indices

        Returns:
            Unpacked indices
        """
        return idxs

    def round(
        self,
        X: torch.Tensor,
        grid: torch.Tensor,
        grid_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find nearest codebook entry for each vector.

        Uses efficient distance computation:
        dist(x, c) = ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2

        Minimizing dist is equivalent to maximizing: 2*x·c - ||c||^2

        Args:
            X: Input vectors [N, codesz]
            grid: Codebook grid [C, codesz]
            grid_norm: Squared norms of grid entries [C]

        Returns:
            vals: Nearest grid entries [N, codesz]
            idx: Indices of nearest entries [N]
        """
        assert X.shape[-1] == self.codesz
        # (2 * X @ grid.T - grid_norm).argmax(-1) finds nearest neighbor
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx
