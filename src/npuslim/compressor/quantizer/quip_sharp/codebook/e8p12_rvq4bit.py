"""
E8P12 RVQ 4-bit Codebook - 4-bit quantization using residual vector quantization.

Uses two E8P codebooks:
1. First pass: quantize to E8P (2 bits)
2. Second pass: quantize residual * scale (2 bits)

Total: 4 bits per weight (2 + 2 bits).

Reference: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12_rvq4bit.py
"""

import torch
from typing import Tuple, Optional

from .base_codebook import BaseCodebook
from .e8p12 import (
    E8P12Codebook,
    _E8P_PACKED_ABS_CACHED,
    _E8P_GRID,
    _PARITY_IDX,
    _get_abs_grid,
)


class E8P12RVQ4BCodebook(BaseCodebook):
    """
    E8P RVQ 4-bit codebook for QuIP#.

    Uses residual vector quantization with two E8P passes:
    - First pass: 2 bits
    - Second pass (residual): 2 bits
    Total: 4 bits per weight
    """

    codesz = 8
    idx_dtype = torch.int64
    packs = 2  # Two passes: init + residual
    pack_out = False
    version = 1
    opt_scale = 1.03  # Same as E8P
    opt_resid_scale = 3.45  # Optimal residual scale for 4-bit

    def __init__(self, inference: bool = False):
        super().__init__(inference=inference)

        # Register packed absolute grid (needed for decompression)
        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED.clone())

        if not inference:
            # Create inner E8P codebook for reuse
            self._e8p = E8P12Codebook(inference=False)

            # Share buffers with inner codebook
            self.register_buffer('grid', _E8P_GRID.clone())
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1) ** 2)

            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            mask = (
                ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) &
                (grid_part[:, :7].min(dim=-1).values >= -0.5)
            )
            grid_part = grid_part[mask]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1) ** 2)

            abs_grid = _get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)

            # Map partial grid to absolute grid
            Xqidx = (2 * grid_part.abs() @ abs_grid.T - abs_grid.norm(dim=-1) ** 2).argmax(-1)
            self.register_buffer('part_abs_map', Xqidx)

            self.register_buffer('bit_map', 2 ** torch.arange(8))

    def round(
        self,
        X: torch.Tensor,
        grid: torch.Tensor,
        grid_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find nearest codebook entry."""
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(
        self,
        X: torch.Tensor,
        parity: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast quantization using partial grid (same as E8P)."""
        X_part = torch.abs(X)

        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]

        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]

        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)

        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = ((roundout < 0) ^ (mask < 0))[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity

        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx

        return vals, idx, err

    def quantize_e8p(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single E8P quantization pass."""
        X_plus = X + 1 / 4
        X_minus = X - 1 / 4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)

        return final_vals, final_idx

    def quantize(
        self,
        X: torch.Tensor,
        return_idx: bool = True,
        resid_scale_override: float = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize vectors using RVQ (two passes).

        Args:
            X: Input tensor [*, 8]
            return_idx: If True, return codebook indices
            resid_scale_override: Override residual scale (-1 for default)

        Returns:
            quantized: Quantized tensor [*, 8]
            idx: Combined indices [*] (init_idx << 16 + resid_idx)
        """
        assert X.shape[-1] == self.codesz

        original_shape = X.shape
        X = X.view(-1, self.codesz)

        # === First pass: initial quantization ===
        init_vals, init_idxs = self.quantize_e8p(X)

        # === Second pass: residual quantization ===
        resid_scale = resid_scale_override if resid_scale_override > 0 else self.opt_resid_scale
        resid = (X - init_vals) * resid_scale
        resid_vals, resid_idxs = self.quantize_e8p(resid)

        # Combine
        final_vals = init_vals + resid_vals / resid_scale
        final_idxs = (init_idxs << 16) + resid_idxs

        final_vals = final_vals.view(original_shape)
        final_idxs = final_idxs.view(original_shape[:-1])

        if return_idx:
            return final_vals, final_idxs
        return final_vals, None

    def by_idxs(self, idxs: torch.Tensor, resid_scale_override: float = -1, **kwargs) -> torch.Tensor:
        """
        Decompress codebook indices to vectors.

        For RVQ, idxs contains both init and residual indices packed together.
        """
        resid_scale = resid_scale_override if resid_scale_override > 0 else self.opt_resid_scale

        m, n = idxs.shape

        # Split into init and residual
        # idxs shape: [m, n] where each element is (init_idx << 16) + resid_idx
        init_idxs = (idxs >> 16).contiguous()
        resid_idxs = (idxs & ((1 << 16) - 1)).contiguous()

        # Decompress init (using Python fallback)
        W_init = self._decompress_e8p(init_idxs)

        # Decompress residual
        W_resid = self._decompress_e8p(resid_idxs) / resid_scale

        return W_init + W_resid

    def _decompress_e8p(self, idxs: torch.Tensor) -> torch.Tensor:
        """Decompress single E8P indices."""
        m, n = idxs.shape
        W = torch.zeros(m, n * self.codesz, dtype=torch.float16, device=idxs.device)

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
                    W[i, j * 8 + k] = val

        return W

    def maybe_pack_idxs(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Pack indices for efficient storage.

        For RVQ, we pack init and residual indices separately.
        Output shape: [m, n * 2] (doubled for init + residual)
        """
        # Split init and residual
        init_idxs = (idxs >> 16).contiguous()
        resid_idxs = (idxs & ((1 << 16) - 1)).contiguous()

        def pack_one(idxs_to_pack: torch.Tensor) -> torch.Tensor:
            m, n = idxs_to_pack.shape
            idxs_reshaped = idxs_to_pack.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2).contiguous()

            abs32 = (
                (idxs_reshaped[:, :, 0, 0] >> 8) +
                ((idxs_reshaped[:, :, 1, 0] >> 8) << 8) +
                ((idxs_reshaped[:, :, 0, 1] >> 8) << 16) +
                ((idxs_reshaped[:, :, 1, 1] >> 8) << 24)
            )

            sign32 = torch.zeros_like(abs32)
            for i in range(4):
                wt = idxs_reshaped[:, :, i % 2, i // 2]
                for j in range(8):
                    sign32 += ((wt >> j) & 1) << (4 * j + i)

            output = (sign32.to(torch.int64) << 32) + abs32
            output = output.reshape(m // 16, 8, n // 8, 4).transpose(1, 2).contiguous()
            return output.view(m, n // 4)

        return torch.concat([pack_one(init_idxs), pack_one(resid_idxs)], dim=-1)
