"""
E8P12 Codebook - 2-bit quantization using E8 lattice.

E8 lattice provides optimal 8-dimensional unit ball packing.
This codebook implements:
- D8^ = D8 + 1/2 intersected with ball of radius sqrt(10) (227 entries)
- Plus 29 entries from vectors with 5 * 3/2 and 3 * 1/2
- Total: 256 base entries * 2^7 sign flips * 2 (±1/4 offset) = 2^16 entries

This gives exactly 2 bits per weight (16 bits / 8 dimensions = 2 bits/dim).

Reference: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12.py
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .base_codebook import BaseCodebook


# Code size for E8 lattice
_E8P_CODESZ = 8


def _get_norm12() -> torch.Tensor:
    """
    Get the 29 elements of norm 12 in E8 + 1/4.

    These are vectors with 5 elements of 3/2 and 3 elements of 1/2.
    """
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ], dtype=torch.float32) / 2


def _get_packed_abs_grid() -> torch.Tensor:
    """
    Generate the packed absolute grid for E8P codebook.

    The grid is stored in a compact format where each 8-element vector
    is packed into a single int32.
    """
    intr = torch.arange(-4, 4, dtype=torch.float32)
    # D8 lattice: all integer vectors with even sum, shifted by 1/2
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    # Filter: sum must be even
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    # Filter: norm squared <= 10
    d8n = d8.norm(dim=-1) ** 2 <= 10
    # Get unique absolute values
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)

    # Add norm 12 elements
    norm12 = _get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)

    # Shuffle and encode into packed format
    shuffle_map = [0, 2, 4, 6, 1, 3, 5, 7]
    cba = cba[:, shuffle_map]
    # Adjust last element based on parity
    cba[:, 7] = cba[:, 7] * (1 - 2 * (cba.sum(1) % 2))
    # Pack into int32
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)

    # Pack 8 x 4-bit values into single int32
    acc = cba[:, 0].clone()
    for i in range(7):
        acc = acc | (cba[:, i + 1] << ((i + 1) * 4))

    return acc


def _get_abs_grid() -> torch.Tensor:
    """Get the unpacked absolute grid (256 x 8)."""
    intr = torch.arange(-4, 4, dtype=torch.float32)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = _get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def _get_full_grid(packed_abs_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Generate the full codebook grid from packed absolute values.

    The full grid has 2^16 entries:
    - 2^8 absolute codes (256 base vectors)
    - 2^8 sign patterns (with parity constraint)

    Returns:
        grid: Full grid [65536, 8]
        grid_idx: Indices [0, 1, ..., 65535]
        parity_idx: Indices where parity bit is 1
    """
    synth_codebook = torch.zeros(1 << 16, 8, dtype=torch.float32)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]

    for c in range(1 << 16):
        signs = c & 255  # Lower 8 bits for signs
        abs_idx = c >> 8  # Upper 8 bits for absolute code

        # Compute parity
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity  # Adjust signs based on parity

        # Decode packed absolute code
        abs_code = packed_abs_grid[abs_idx].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if (signs >> ii) & 1:
                synth_codebook[c, i] *= -1

        # Add/subtract 1/4 based on parity
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25

    return synth_codebook, torch.arange(1 << 16), parity_idx


# Precompute grids at module load time
_E8P_PACKED_ABS_CACHED = _get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = _get_full_grid(_E8P_PACKED_ABS_CACHED)


class E8P12Codebook(BaseCodebook):
    """
    E8P 2-bit codebook for QuIP#.

    Uses E8 lattice with 2^16 codebook entries, achieving 2 bits per weight.
    """

    codesz = _E8P_CODESZ
    idx_dtype = torch.int64
    packs = 4
    pack_out = False
    version = 1
    opt_scale = 1.03  # Optimal scale determined empirically

    def __init__(self, inference: bool = False):
        super().__init__(inference=inference)

        # Register packed absolute grid (needed for decompression)
        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED.clone())

        if not inference:
            # Register full grid for quantization
            self.register_buffer('grid', _E8P_GRID.clone())
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1) ** 2)

            # Create partial grid for faster quantization
            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            mask = (
                ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) &
                (grid_part[:, :7].min(dim=-1).values >= -0.5)
            )
            grid_part = grid_part[mask]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1) ** 2)

            # Map partial grid to absolute grid
            abs_grid = _get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)
            _, part_abs_map = self.round(grid_part.abs(), abs_grid, abs_grid.norm(dim=-1) ** 2)
            self.register_buffer('part_abs_map', part_abs_map)

            # Bit map for sign encoding
            self.register_buffer('bit_map', 2 ** torch.arange(8))

    def fast_quantize_part(
        self,
        X: torch.Tensor,
        parity: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast quantization using partial grid.

        Args:
            X: Input tensor [N, 8] (already offset by ±1/4)
            parity: Whether to use parity=1 encoding

        Returns:
            vals: Quantized values [N, 8]
            idx: Codebook indices [N]
            err: Quantization error [N]
        """
        X_part = torch.abs(X)

        # Adjust last element for odd sign count
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]

        # Create sign mask
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]

        # Find nearest in partial grid
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)

        # Encode to codebook index
        abs_idx = self.part_abs_map[Xqidx]

        # Compute sign mask
        sign_mask = ((roundout < 0) ^ (mask < 0))[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity

        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx

        return vals, idx, err

    def quantize(
        self,
        X: torch.Tensor,
        return_idx: bool = True,
        resid_scale_override: float = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize vectors to E8P codebook.

        Args:
            X: Input tensor [*, 8] - last dim must be divisible by codesz
            return_idx: If True, return codebook indices
            resid_scale_override: Not used for E8P

        Returns:
            quantized: Quantized tensor [*, 8]
            idx: Codebook indices [*] (if return_idx=True)
        """
        assert X.shape[-1] == self.codesz, f"Expected last dim {self.codesz}, got {X.shape[-1]}"

        original_shape = X.shape
        X = X.view(-1, self.codesz)

        # Try both D8^ - 1/4 and D8^ + 1/4
        X_plus = X + 1 / 4
        X_minus = X - 1 / 4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(X_minus, False)

        # Choose better option
        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)

        final_vals = final_vals.view(original_shape)
        final_idx = final_idx.view(original_shape[:-1])

        if return_idx:
            return final_vals, final_idx
        return final_vals, None

    def by_idxs(self, idxs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decompress codebook indices to vectors.

        Args:
            idxs: Codebook indices

        Returns:
            Decompressed tensor
        """
        # For now, use Python implementation
        # TODO: Add CUDA kernel for faster decompression
        m, n = idxs.shape

        # Decode from packed format
        W_decompressed = torch.zeros(m, n * self.codesz, dtype=torch.float16, device=idxs.device)

        shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
        grid_packed_abs = self.grid_packed_abs

        for i in range(m):
            for j in range(n):
                c = idxs[i, j].item()
                signs = c & 255
                abs_idx = c >> 8

                parity = 0
                for k in range(8):
                    parity = parity ^ ((signs >> k) & 1)
                signs = signs ^ parity

                abs_code = grid_packed_abs[abs_idx].item()
                for k in range(8):
                    ii = shuffle_map[k]
                    val = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
                    if (signs >> ii) & 1:
                        val *= -1
                    if parity:
                        val -= 0.25
                    else:
                        val += 0.25
                    W_decompressed[i, j * 8 + k] = val

        return W_decompressed

    def maybe_pack_idxs(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Pack indices for efficient storage.

        Packs 4 x 16-bit indices into 2 x 32-bit values.
        """
        m, n = idxs.shape
        # Reshape for packing: [m/2, 2, n/2, 2]
        idxs = idxs.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2).contiguous()

        # Pack absolute indices (upper 8 bits)
        abs32 = (
            (idxs[:, :, 0, 0] >> 8) +
            ((idxs[:, :, 1, 0] >> 8) << 8) +
            ((idxs[:, :, 0, 1] >> 8) << 16) +
            ((idxs[:, :, 1, 1] >> 8) << 24)
        )

        # Pack sign bits (lower 8 bits)
        sign32 = torch.zeros_like(abs32)
        for i in range(4):
            wt = idxs[:, :, i % 2, i // 2]
            for j in range(8):
                sign32 += ((wt >> j) & 1) << (4 * j + i)

        output = (sign32.to(torch.int64) << 32) + abs32
        output = output.reshape(m // 16, 8, n // 8, 4).transpose(1, 2).contiguous()

        return output.view(m, n // 4)
