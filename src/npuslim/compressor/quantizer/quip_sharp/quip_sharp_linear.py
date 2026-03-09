"""
QuIP# Linear Layer for inference.

This layer stores quantized weights as codebook indices and performs
on-the-fly decompression during forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional

from .codebook import get_codebook
from .utils.hadamard import matmul_hadU_cuda, get_hadK


class QuIPSharpLinear(nn.Module):
    """
    Quantized linear layer for QuIP# inference.

    Stores weights as packed codebook indices and performs:
    1. Apply SU (input sign vector)
    2. Hadamard transform on input
    3. Decompress weights from codebook indices
    4. Matrix multiplication
    5. Hadamard transform on output
    6. Apply SV (output sign vector)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codebook_name: str = "E8P12",
        codebook_version: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.codebook_name = codebook_name
        self.codebook_version = codebook_version

        # Get codebook (inference mode - smaller memory)
        self.codebook = get_codebook(codebook_name, inference=True)
        self.codesz = self.codebook.codesz

        # Validate dimensions
        assert in_features % self.codesz == 0, \
            f"in_features ({in_features}) must be divisible by codesz ({self.codesz})"

        # Register buffers
        # Qidxs: packed codebook indices
        self.register_buffer(
            "Qidxs",
            torch.zeros(out_features, in_features // self.codesz, dtype=torch.int64),
        )

        # SU: input dimension signs
        self.register_buffer(
            "SU",
            torch.ones(in_features, dtype=torch.float16),
        )

        # SV: output dimension signs (includes Wscale)
        self.register_buffer(
            "SV",
            torch.ones(out_features, dtype=torch.float16),
        )

        # Optional bias
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

        # Hadamard matrix (cached, not saved)
        hadK, K = get_hadK(in_features)
        if hadK is not None:
            self.register_buffer("hadK", hadK, persistent=False)
        else:
            self.hadK = None

        # Lazy initialization
        self._decompressed_weight = None
        self._initialized = False

    def pack(
        self,
        Qidxs: torch.Tensor,
        SU: torch.Tensor,
        SV: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Pack quantized weights into the layer.

        Args:
            Qidxs: Codebook indices [out_features, in_features // codesz]
            SU: Input signs [in_features]
            SV: Output signs [out_features]
            bias: Optional bias [out_features]
        """
        self.Qidxs.copy_(Qidxs)
        self.SU.copy_(SU.to(torch.float16))
        self.SV.copy_(SV.to(torch.float16))

        if bias is not None:
            self.bias.copy_(bias.to(torch.float16))

        self._initialized = True
        self._decompressed_weight = None  # Clear cache

    def _decompress_weights(self, resid_scale_override: float = -1) -> torch.Tensor:
        """Decompress weights from codebook indices."""
        if self._decompressed_weight is not None:
            return self._decompressed_weight

        # Decompress using codebook
        W = self.codebook.by_idxs(self.Qidxs, resid_scale_override=resid_scale_override)

        # Scale by SV
        W = W * self.SV.unsqueeze(1)

        self._decompressed_weight = W.to(torch.float16)
        return self._decompressed_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly decompression.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Save original shape and dtype
        original_shape = x.shape
        original_dtype = x.dtype
        x = x.view(-1, self.in_features)

        # Step 1: Apply SU (input signs)
        x = x * self.SU.to(x.device, x.dtype)

        # Step 2: Hadamard transform on input
        x = matmul_hadU_cuda(x.float(), self.hadK)

        # Step 3: Decompress weights and compute
        W = self._decompress_weights()
        x = x.to(W.device) @ W.T

        # Step 4: Hadamard transform on output
        x = matmul_hadU_cuda(x.float(), self.hadK)

        # Step 5: Apply SV (output signs) - already included in W
        # x = x * self.SV.to(x.device, x.dtype)

        # Add bias if present
        if self.bias is not None:
            x = x + self.bias.to(x.device, x.dtype)

        # Restore shape and dtype
        x = x.view(*original_shape[:-1], self.out_features)
        x = x.to(original_dtype)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"codebook={self.codebook_name}, bias={self.bias is not None}"
        )
