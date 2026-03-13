"""Unit tests for GPTQ column-wise (input dimension) int32 weight packing.

Tests verify the pack/unpack roundtrip for column-wise packing format
expected by vLLM-Ascend.

vLLM-Ascend expects:
    weight:       [outfeatures, infeatures // 8] as int32 (packed int4 along input dim)
    weight_scale: [outfeatures, num_groups] as bfloat16
    weight_offset: [outfeatures, num_groups] as bfloat16
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pack_int4_to_int32_colwise(weight_int8: torch.Tensor, pack_factor: int = 8) -> torch.Tensor:
    """Pack int4 weights (stored as int8) into int32 along input (column) dimension.

    Args:
        weight_int8: Signed int8 tensor with values in [-8, 7], shape [out, in]
        pack_factor: Number of int4 values per int32 (default 8)

    Returns:
        Packed int32 tensor with shape [out, in // pack_factor]
    """
    outfeatures, infeatures = weight_int8.shape
    assert infeatures % pack_factor == 0, (
        f"infeatures {infeatures} not divisible by pack_factor {pack_factor}"
    )

    # Convert signed [-8, 7] to unsigned [0, 15]
    unsigned = (weight_int8 + 8).to(torch.uint8)

    # Pack into int32 along input dimension (columns)
    packed = torch.zeros((outfeatures, infeatures // pack_factor), dtype=torch.int32)
    for i in range(pack_factor):
        packed |= (unsigned[:, i::pack_factor].to(torch.int32) << (4 * i))

    return packed


def unpack_int32_to_int4_colwise(packed: torch.Tensor, pack_factor: int = 8) -> torch.Tensor:
    """Unpack int4 weights from int32 to int8 (column-wise unpacking).

    Args:
        packed: Packed int32 tensor with shape [out, in // pack_factor]
        pack_factor: Number of int4 values per int32 (default 8)

    Returns:
        Unpacked int8 tensor with shape [out, in] and values in [-8, 7]
    """
    outfeatures, packed_in = packed.shape
    infeatures = packed_in * pack_factor

    mask = 0xF
    offset = 8

    unpacked = torch.zeros((outfeatures, infeatures), dtype=torch.int8)
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = ((packed >> (4 * i)) & mask).to(torch.int8) - offset

    return unpacked


class TestColumnWisePacking:
    """Tests for column-wise int32 weight packing/unpacking."""

    def test_roundtrip_basic(self):
        """Test basic pack/unpack roundtrip."""
        original = torch.randint(-8, 8, (32, 16), dtype=torch.int8)
        packed = pack_int4_to_int32_colwise(original)
        unpacked = unpack_int32_to_int4_colwise(packed)
        assert torch.all(original == unpacked), "Pack/unpack mismatch!"

    def test_roundtrip_edge_values(self):
        """Test with edge values -8 and 7."""
        # Create matrix with edge values
        original = torch.tensor([[-8, -1, 0, 1, 7, 3, -5, 2]], dtype=torch.int8)
        original = original.repeat(8, 1)  # [8, 8]
        packed = pack_int4_to_int32_colwise(original)
        unpacked = unpack_int32_to_int4_colwise(packed)
        assert torch.all(original == unpacked), "Edge values mismatch!"

    def test_roundtrip_large_matrix(self):
        """Test with realistic weight matrix size."""
        torch.manual_seed(42)
        original = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8)
        packed = pack_int4_to_int32_colwise(original)
        unpacked = unpack_int32_to_int4_colwise(packed)
        assert torch.all(original == unpacked), "Large matrix mismatch!"

    def test_storage_compression(self):
        """Verify correct storage shape for column-wise packing."""
        original = torch.randint(-8, 8, (4096, 4096), dtype=torch.int8)
        packed = pack_int4_to_int32_colwise(original)

        # Column-wise: [4096, 4096] -> [4096, 512]
        assert packed.shape == (4096, 512), f"Expected (4096, 512), got {packed.shape}"

        # Verify compression ratio: 8 int4 -> 1 int32
        original_bytes = original.numel() * original.element_size()  # 16MB
        packed_bytes = packed.numel() * packed.element_size()  # 8MB
        assert packed_bytes == original_bytes // 2, "Expected 2x compression"

    def test_all_possible_values(self):
        """Test all 16 possible int4 values."""
        all_values = torch.arange(-8, 8, dtype=torch.int8).reshape(1, 16)
        original = all_values.repeat(8, 1)  # [8, 16]

        packed = pack_int4_to_int32_colwise(original)
        unpacked = unpack_int32_to_int4_colwise(packed)

        assert torch.all(original == unpacked), "Not all values preserved!"


class TestGPTQQuantLinearColumnWise:
    """Tests for GPTQQuantLinear with column-wise packing."""

    def test_buffer_shapes_column_wise(self):
        """Test that Ascend format buffers have correct column-wise shapes."""
        from npuslim.compressor.quantizer.gptq.gptq_linear import GPTQQuantLinear
        from npuslim.utils.backend import bh

        original_backend = bh.name
        bh.name = "npu"

        try:
            outfeatures = 256
            infeatures = 512
            group_size = 128

            linear = GPTQQuantLinear(
                bits=4,
                group_size=group_size,
                infeatures=infeatures,
                outfeatures=outfeatures,
                bias=False,
            )

            # Column-wise: weight shape should be [out, in//8]
            expected_weight_shape = (outfeatures, infeatures // 8)
            assert linear.weight.shape == expected_weight_shape, (
                f"Expected weight shape {expected_weight_shape}, "
                f"got {linear.weight.shape}"
            )

            # Scale/offset: [out, num_groups]
            num_groups = (infeatures + group_size - 1) // group_size
            expected_scale_shape = (outfeatures, num_groups)
            assert linear.weight_scale.shape == expected_scale_shape, (
                f"Expected scale shape {expected_scale_shape}, "
                f"got {linear.weight_scale.shape}"
            )
            assert linear.weight_offset.shape == expected_scale_shape, (
                f"Expected offset shape {expected_scale_shape}, "
                f"got {linear.weight_offset.shape}"
            )
        finally:
            bh.name = original_backend

    def test_pack_forward_roundtrip(self):
        """Test that pack -> forward produces correct dequantized output."""
        from npuslim.compressor.quantizer.gptq.gptq_linear import GPTQQuantLinear
        from npuslim.utils.backend import bh

        original_backend = bh.name
        bh.name = "npu"

        try:
            outfeatures, infeatures = 64, 128
            group_size = 32

            # Create a simple linear layer with known weights
            torch.manual_seed(42)
            linear = torch.nn.Linear(infeatures, outfeatures, bias=False)
            original_weight = linear.weight.data.clone()

            # Create quantization parameters (uniform for simplicity)
            num_groups = infeatures // group_size
            scales = torch.ones(outfeatures, num_groups) * 0.1
            zeros = torch.zeros(outfeatures, num_groups)

            # Create GPTQQuantLinear and pack
            quant_linear = GPTQQuantLinear(
                bits=4,
                group_size=group_size,
                infeatures=infeatures,
                outfeatures=outfeatures,
                bias=False,
            )
            quant_linear.pack(linear, scales, zeros)

            # Verify packed weight shape
            assert quant_linear.weight.shape == (outfeatures, infeatures // 8), (
                f"Expected ({outfeatures}, {infeatures // 8}), "
                f"got {quant_linear.weight.shape}"
            )

            # Run forward pass
            x = torch.randn(2, infeatures)
            output = quant_linear(x)

            # Verify output shape
            assert output.shape == (2, outfeatures), (
                f"Expected (2, {outfeatures}), got {output.shape}"
            )
        finally:
            bh.name = original_backend

    def test_pack_forward_matches_expected(self):
        """Test that pack -> forward matches expected quantized matmul."""
        from npuslim.compressor.quantizer.gptq.gptq_linear import GPTQQuantLinear
        from npuslim.utils.backend import bh

        original_backend = bh.name
        bh.name = "npu"

        try:
            outfeatures, infeatures = 32, 64
            group_size = 16

            # Create weight matrix with small values for predictable quantization
            torch.manual_seed(123)
            weight_fp = torch.randn(outfeatures, infeatures) * 0.05

            linear = torch.nn.Linear(infeatures, outfeatures, bias=False)
            linear.weight.data = weight_fp.clone()

            # Simple uniform quantization: scale=0.1, zero=0
            num_groups = infeatures // group_size
            scales = torch.ones(outfeatures, num_groups) * 0.1
            zeros = torch.zeros(outfeatures, num_groups)

            # Create and pack
            quant_linear = GPTQQuantLinear(
                bits=4,
                group_size=group_size,
                infeatures=infeatures,
                outfeatures=outfeatures,
                bias=False,
            )
            quant_linear.pack(linear, scales, zeros)

            # Run forward
            x = torch.randn(4, infeatures)
            output = quant_linear(x)

            # Manual reference computation:
            # 1. Quantize weight: q = round((w + zero*scale) / scale) - offset
            # 2. Dequantize: w_q = (q + offset) * scale
            # 3. Matmul: y = x @ w_q.T
            signed_offset = 8
            g_idx = torch.arange(infeatures) // group_size
            current_scales = scales[:, g_idx]  # [out, in]

            q = torch.round((weight_fp + zeros[:, g_idx] * scales[:, g_idx]) / current_scales) - signed_offset
            q = q.clamp(-8, 7).to(torch.int8)
            w_dequant = (q.float() + 0) * current_scales  # offset=0 in our pack

            expected_output = torch.matmul(x, w_dequant.T)

            # Allow small numerical differences (quantization introduces ~1% error)
            # max diff should be < 0.01 for 4-bit quantization
            max_diff = (output - expected_output).abs().max()
            assert max_diff < 0.01, f"Output mismatch! max diff: {max_diff}"
        finally:
            bh.name = original_backend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
