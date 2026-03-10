"""
Integration tests for GPTQ GPU to NPU converter.

These tests verify the conversion functionality with mock data.
For full integration tests with real models, network access is required.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tools.convert.gptq_gpu_to_npu import (
    load_quant_config,
    unpack_gptq_weights,
    pack_ascend_weights,
)


class TestLoadQuantConfig:
    """Tests for load_quant_config function."""

    def test_load_valid_config(self):
        """Test loading a valid quantization config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model_type": "qwen2",
                "quantization_config": {
                    "bits": 4,
                    "group_size": 128,
                    "desc_act": True,
                    "static_groups": True,
                    "sym": True,
                    "quant_method": "gptq",
                }
            }
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            quant_cfg = load_quant_config(tmpdir)
            assert quant_cfg["bits"] == 4
            assert quant_cfg["group_size"] == 128
            assert quant_cfg["desc_act"] == True

    def test_missing_quantization_config(self):
        """Test error when quantization_config is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model_type": "qwen2"}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            with pytest.raises(ValueError, match="No quantization_config found"):
                load_quant_config(tmpdir)


class TestUnpackGPTQWeights:
    """Tests for unpack_gptq_weights function."""

    def test_unpack_shapes(self):
        """Test that unpacking produces correct shapes."""
        outfeatures, infeatures = 128, 256
        bits = 4
        group_size = 128
        num_groups = infeatures // group_size

        qweight = torch.randint(0, 2**31, (infeatures // 8, outfeatures), dtype=torch.int32)
        qzeros = torch.randint(0, 2**31, (num_groups, outfeatures // 8), dtype=torch.int32)
        scales = torch.randn(num_groups, outfeatures)
        g_idx = torch.arange(infeatures, dtype=torch.int32) // group_size

        weight, zeros, out_scales = unpack_gptq_weights(
            qweight, qzeros, scales, g_idx, bits, group_size
        )

        assert weight.shape == (infeatures, outfeatures)
        assert zeros.shape == (num_groups, outfeatures)
        assert out_scales.shape == (num_groups, outfeatures)

    def test_unpack_value_range(self):
        """Test that unpacked weights are in valid 4-bit range."""
        outfeatures, infeatures = 64, 128
        bits = 4
        group_size = 64
        num_groups = infeatures // group_size

        qweight = torch.randint(0, 2**31, (infeatures // 8, outfeatures), dtype=torch.int32)
        qzeros = torch.randint(0, 2**31, (num_groups, outfeatures // 8), dtype=torch.int32)
        scales = torch.randn(num_groups, outfeatures)
        g_idx = torch.arange(infeatures, dtype=torch.int32) // group_size

        weight, _, _ = unpack_gptq_weights(
            qweight, qzeros, scales, g_idx, bits, group_size
        )

        assert weight.min() >= 0
        assert weight.max() < 16  # 4-bit range [0, 15]


class TestPackAscendWeights:
    """Tests for pack_ascend_weights function."""

    def test_pack_shapes(self):
        """Test that packing produces correct shapes."""
        infeatures, outfeatures = 256, 128
        bits = 4
        group_size = 128
        num_groups = infeatures // group_size

        weight = torch.randint(0, 16, (infeatures, outfeatures), dtype=torch.int32)
        zeros = torch.randn(num_groups, outfeatures)
        scales = torch.randn(num_groups, outfeatures).abs()
        g_idx = torch.arange(infeatures, dtype=torch.int32) // group_size

        packed_weight, weight_scale, weight_offset = pack_ascend_weights(
            weight, zeros, scales, g_idx, bits, group_size
        )

        assert packed_weight.shape == (outfeatures // 2, infeatures)
        assert weight_scale.shape == (outfeatures, num_groups)
        assert weight_offset.shape == (outfeatures, num_groups)

    def test_pack_dtypes(self):
        """Test that packed tensors have correct dtypes."""
        infeatures, outfeatures = 128, 64
        bits = 4
        group_size = 64
        num_groups = infeatures // group_size

        weight = torch.randint(0, 16, (infeatures, outfeatures), dtype=torch.int32)
        zeros = torch.randn(num_groups, outfeatures)
        scales = torch.randn(num_groups, outfeatures).abs()
        g_idx = torch.arange(infeatures, dtype=torch.int32) // group_size

        packed_weight, weight_scale, weight_offset = pack_ascend_weights(
            weight, zeros, scales, g_idx, bits, group_size
        )

        assert packed_weight.dtype == torch.int8
        assert weight_scale.dtype == torch.bfloat16
        assert weight_offset.dtype == torch.bfloat16
