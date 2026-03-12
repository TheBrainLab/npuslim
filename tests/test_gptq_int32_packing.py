"""Unit tests for GPTQ int32 weight packing/unpacking.

Tests verify the pack/unpack roundtrip and compatibility with upstream vllm-ascend.
"""

import torch
import pytest


def pack_int4_to_int32(weight_int8: torch.Tensor, pack_factor: int = 8) -> torch.Tensor:
    """Pack int4 weights (stored as int8) into int32.

    Args:
        weight_int8: Signed int8 tensor with values in [-8, 7]
        pack_factor: Number of int4 values per int32 (default 8)

    Returns:
        Packed int32 tensor with shape [out // pack_factor, in]
    """
    outfeatures, infeatures = weight_int8.shape
    assert outfeatures % pack_factor == 0, (
        f"outfeatures {outfeatures} not divisible by pack_factor {pack_factor}"
    )

    # Convert signed [-8, 7] to unsigned [0, 15]
    unsigned = (weight_int8 + 8).to(torch.uint8)

    # Pack into int32
    packed = torch.zeros((outfeatures // pack_factor, infeatures), dtype=torch.int32)
    for i in range(pack_factor):
        packed |= (unsigned[i::pack_factor, :].to(torch.int32) << (4 * i))

    return packed


def unpack_int32_to_int4(packed: torch.Tensor, pack_factor: int = 8) -> torch.Tensor:
    """Unpack int4 weights from int32 to int8.

    Args:
        packed: Packed int32 tensor with shape [out // pack_factor, in]
        pack_factor: Number of int4 values per int32 (default 8)

    Returns:
        Unpacked int8 tensor with shape [out, in] and values in [-8, 7]
    """
    packed_out, infeatures = packed.shape
    outfeatures = packed_out * pack_factor

    mask = 0xF
    offset = 8

    unpacked = torch.zeros((outfeatures, infeatures), dtype=torch.int8)
    for i in range(pack_factor):
        unpacked[i::pack_factor, :] = ((packed >> (4 * i)) & mask).to(torch.int8) - offset

    return unpacked


class TestInt32Packing:
    """Tests for int32 weight packing/unpacking."""

    def test_roundtrip_basic(self):
        """Test basic pack/unpack roundtrip."""
        original = torch.randint(-8, 8, (16, 32), dtype=torch.int8)
        packed = pack_int4_to_int32(original)
        unpacked = unpack_int32_to_int4(packed)
        assert torch.all(original == unpacked), "Pack/unpack mismatch!"

    def test_roundtrip_edge_values(self):
        """Test with edge values -8 and 7."""
        original = torch.tensor([[-8, -1, 0, 1, 7]] * 8, dtype=torch.int8)
        packed = pack_int4_to_int32(original)
        unpacked = unpack_int32_to_int4(packed)
        assert torch.all(original == unpacked), "Edge values mismatch!"

    def test_roundtrip_large_matrix(self):
        """Test with realistic weight matrix size."""
        original = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8)
        packed = pack_int4_to_int32(original)
        unpacked = unpack_int32_to_int4(packed)
        assert torch.all(original == unpacked), "Large matrix mismatch!"

    def test_storage_compression(self):
        """Verify correct storage shape."""
        original = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8)
        packed = pack_int4_to_int32(original)

        # Shape check: [4096, 4096] -> [512, 4096]
        assert packed.shape == (512, 4096), f"Unexpected shape: {packed.shape}"

        # Verify compression ratio
        original_bytes = original.numel() * original.element_size()  # 16MB
        packed_bytes = packed.numel() * packed.element_size()  # 8MB
        assert packed_bytes == original_bytes // 2, "Expected 2x compression"

    def test_all_possible_values(self):
        """Test all 16 possible int4 values."""
        # Create matrix with all values -8 to 7
        all_values = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
        # Repeat to get multiple of 8 rows
        original = all_values.repeat(8, 1)

        packed = pack_int4_to_int32(original)
        unpacked = unpack_int32_to_int4(packed)

        assert torch.all(original == unpacked), "Not all values preserved!"

    def test_upstream_unpack_compatibility(self):
        """Test that our pack is compatible with upstream unpack_from_int32."""
        try:
            from vllm_ascend.quantization.methods.w4a16 import unpack_from_int32
        except ImportError:
            pytest.skip("vllm_ascend not available")

        original = torch.randint(-8, 8, (16, 32), dtype=torch.int8)
        packed = pack_int4_to_int32(original)

        # Use upstream unpack with packed_dim=0 (output dimension packed)
        unpacked = unpack_from_int32(
            weight=packed,
            shape=torch.Size([16, 32]),
            num_bits=4,
            packed_dim=0,
        )
        assert torch.all(original == unpacked), "Upstream unpack mismatch!"

    def test_upstream_unpack_large_matrix(self):
        """Test upstream unpack compatibility with larger matrix."""
        try:
            from vllm_ascend.quantization.methods.w4a16 import unpack_from_int32
        except ImportError:
            pytest.skip("vllm_ascend not available")

        original = torch.randint(-8, 8, (1024, 2048), dtype=torch.int8)
        packed = pack_int4_to_int32(original)

        unpacked = unpack_from_int32(
            weight=packed,
            shape=torch.Size([1024, 2048]),
            num_bits=4,
            packed_dim=0,
        )
        assert torch.all(original == unpacked), "Large upstream unpack mismatch!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
