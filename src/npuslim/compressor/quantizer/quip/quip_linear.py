"""
QuIP Quantized Linear Layer.

Supports both minmax (with zero) and rms (without zero) quantization modes.
Uses seed-based Butterfly matrix regeneration for preprocessing parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from npuslim.compressor.quantizer.quip.quip_module import (
    ButterflyMode,
    gen_rand_ortho_butterfly,
    gen_rand_ortho_butterfly_noblock,
    gen_rand_ortho_butterfly_nopermute,
    rand_ortho_matrix,
    mul_ortho_butterfly,
)


class QuIPLinear(nn.Module):
    """
    QuIP quantized linear layer.

    Key differences from GPTQQuantLinear:
    - No g_idx (no grouping)
    - No qzeros (zeros stored directly for minmax, or None for rms)
    - Stores preprocessing parameters (scaleWH, proj_seeds) for postproc
    """

    def __init__(
        self,
        bits: int,
        infeatures: int,
        outfeatures: int,
        has_zero: bool = True,
        bias: bool = False,
        proj_mode: int = 2,
    ):
        """
        Args:
            bits: Quantization bits (2, 3, 4, 8)
            infeatures: Input dimension
            outfeatures: Output dimension
            has_zero: Whether to use zero points (True for minmax, False for rms)
            bias: Whether to include bias
            proj_mode: Butterfly projection mode (0-3)
        """
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")

        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.has_zero = has_zero
        self.maxq = 2**self.bits - 1
        self.proj_mode = proj_mode

        # Packed weights: each int32 holds multiple quantized values
        # Shape: [infeatures // 32 * bits, outfeatures]
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )

        # Quantization parameters
        if has_zero:
            # minmax mode: per-channel scale and zero
            self.register_buffer("scales", torch.zeros((outfeatures, 1), dtype=torch.float16))
            self.register_buffer("zeros", torch.zeros((outfeatures, 1), dtype=torch.float16))
        else:
            # rms mode: single scalar scale, no zero
            self.register_buffer("scales", torch.zeros(1, dtype=torch.float16))
            self.zeros = None

        # Preprocessing parameters for postproc
        self.register_buffer("scaleWH", torch.zeros(infeatures, dtype=torch.float32))
        self.register_buffer("proj_seed_u", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("proj_seed_v", torch.tensor(0, dtype=torch.int64))

        # Cached Butterfly matrices (regenerated on first forward if needed)
        self._cached_projU = None
        self._cached_projV = None

        if bias:
            self.register_buffer("bias", torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None

        # Unpacking factor for bit extraction
        if self.bits in [2, 4, 8]:
            self.register_buffer(
                "wf",
                torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0),
                persistent=False,
            )
        elif self.bits == 3:
            self.register_buffer(
                "wf",
                torch.tensor(
                    [
                        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                        [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                        [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                    ],
                    dtype=torch.int32,
                ).reshape(1, 3, 12),
                persistent=False,
            )

    def post_init(self):
        """Post-initialization hook (for compatibility)."""
        pass

    def _generate_butterfly_from_seed(self, seed: int, size: int) -> torch.Tensor:
        """
        Regenerate Butterfly matrix from seed.

        Args:
            seed: Random seed used to generate the matrix
            size: Matrix dimension

        Returns:
            Dense orthogonal matrix [size, size]
        """
        # Map mode index to ButterflyMode
        mode_map = {
            0: ButterflyMode.BUTTERFLY_PERMUTE,
            1: ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK,
            2: ButterflyMode.BUTTERFLY_NOPERMUTE,
            3: ButterflyMode.RANDOM_ORTHO,
        }
        mode = mode_map.get(self.proj_mode, ButterflyMode.BUTTERFLY_NOPERMUTE)

        # Set seed and regenerate (must set both torch and numpy/scipy seeds)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))  # scipy uses numpy's random state

        if mode == ButterflyMode.BUTTERFLY_PERMUTE:
            B = mul_ortho_butterfly(gen_rand_ortho_butterfly(size), torch.eye(size))
        elif mode == ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK:
            B = mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(size), torch.eye(size))
        elif mode == ButterflyMode.BUTTERFLY_NOPERMUTE:
            B = mul_ortho_butterfly(gen_rand_ortho_butterfly_nopermute(size), torch.eye(size))
        elif mode == ButterflyMode.RANDOM_ORTHO:
            B = rand_ortho_matrix(size)
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

        return B.to(torch.float32)

    def _unpack_weights(self) -> torch.Tensor:
        """
        Unpack integer weights from qweight.

        Returns:
            Integer weight tensor [infeatures, outfeatures]
        """
        if self.bits in [2, 4, 8]:
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int8)
            weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        elif self.bits == 3:
            weight = self.qweight.reshape(
                self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
            )
        else:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")

        # Reshape to [infeatures, outfeatures]
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        return weight

    def _dequantize(self, weight_int: torch.Tensor) -> torch.Tensor:
        """
        Dequantize integer weights to float.

        Args:
            weight_int: Integer weights [infeatures, outfeatures]

        Returns:
            Dequantized weights [outfeatures, infeatures]
        """
        if self.has_zero and self.zeros is not None:
            # minmax: w = scale * (q - zero)
            # scales/zeros are [outfeatures, 1], weight is [infeatures, outfeatures]
            w = self.scales.T * (weight_int.float() - self.zeros.T)
        else:
            # rms: w = (q / maxq * 2 - 1) * scale
            w = (weight_int.float() / self.maxq * 2 - 1) * self.scales
            w = w.T  # [outfeatures, infeatures]

        return w.to(torch.float32)

    def _postproc(self, w: torch.Tensor) -> torch.Tensor:
        """
        Apply postprocessing: reverse preprocessing transformations.

        preproc: W' = U @ (W * scaleWH) @ V.T
        postproc: W = V @ (U.T @ W') @ scaleWH_inv

        Args:
            w: Dequantized weights [outfeatures, infeatures]

        Returns:
            Postprocessed weights [outfeatures, infeatures]
        """
        # Regenerate Butterfly matrices from seeds (with caching)
        if self._cached_projU is None:
            seed_u = int(self.proj_seed_u.item())
            self._cached_projU = self._generate_butterfly_from_seed(
                seed_u, self.outfeatures
            ).to(w.device)
        if self._cached_projV is None:
            seed_v = int(self.proj_seed_v.item())
            self._cached_projV = self._generate_butterfly_from_seed(
                seed_v, self.infeatures
            ).to(w.device)

        U = self._cached_projU
        V = self._cached_projV
        scaleWH = self.scaleWH.to(w.device)

        # Reverse transformation
        # W = V @ (U.T @ W') @ diag(1/scaleWH)
        # Note: w is [out, in], so we need to be careful with matrix ops
        w = U.T @ w @ V
        w = w / scaleWH.unsqueeze(0).clamp(min=1e-8)

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: unpack -> dequantize -> postproc -> linear
        """
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        # Ensure wf is on correct device
        if self.wf.device != self.qweight.device:
            self.wf = self.wf.to(self.qweight.device)

        # 1. Unpack integer weights
        weight_int = self._unpack_weights()

        # 2. Dequantize
        w = self._dequantize(weight_int)

        # 3. Postproc (reverse preprocessing)
        w = self._postproc(w)

        # 4. Linear (convert weight and bias to input dtype)
        bias = self.bias.to(x_dtype) if self.bias is not None else None
        out = F.linear(x, w.to(x_dtype), bias)
        out = out.reshape(out_shape)

        return out

    def pack(
        self,
        w_int: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        scaleWH: torch.Tensor,
        proj_seed_u: int,
        proj_seed_v: int,
        bias: torch.Tensor | None = None,
    ):
        """
        Pack quantized weights and parameters.

        Args:
            w_int: Integer weights [outfeatures, infeatures] in range [0, maxq]
            scales: Quantization scales
            zeros: Zero points (None for rms mode)
            scaleWH: Preprocessing scale [infeatures]
            proj_seed_u: Seed for projU
            proj_seed_v: Seed for projV
            bias: Optional bias tensor
        """
        # Store quantization parameters
        self.scales.copy_(scales)
        if zeros is not None and self.zeros is not None:
            self.zeros.copy_(zeros)
        self.scaleWH.copy_(scaleWH)
        self.proj_seed_u.fill_(proj_seed_u)
        self.proj_seed_v.fill_(proj_seed_v)
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)

        # Pack integer weights into qweight
        # w_int is [outfeatures, infeatures], need to transpose and pack
        w_int = w_int.t().contiguous()  # [infeatures, outfeatures]
        w_int_np = w_int.numpy().astype(np.uint32)

        qweight = np.zeros(
            (w_int_np.shape[0] // 32 * self.bits, w_int_np.shape[1]), dtype=np.uint32
        )

        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= w_int_np[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i))
                i += 10
                qweight[row] |= w_int_np[i] << 30
                row += 1
                qweight[row] |= (w_int_np[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= w_int_np[i] << 31
                row += 1
                qweight[row] |= (w_int_np[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")

        self.qweight.copy_(torch.from_numpy(qweight.astype(np.int32)))

    def clear_cache(self):
        """Clear cached Butterfly matrices to free memory."""
        self._cached_projU = None
        self._cached_projV = None
