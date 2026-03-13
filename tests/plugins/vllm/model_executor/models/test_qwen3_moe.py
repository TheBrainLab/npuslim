"""Tests for Qwen3 MoE model patch."""
import pytest
from unittest.mock import MagicMock, patch


class TestQwen3MoEModelPatch:
    """Test Qwen3 MoE model patching."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create mock vLLM config."""
        config = MagicMock()
        config.quant_config = MagicMock()
        config.quant_config.quant_description = {"group_size": 128}
        return config

    def test_patch_function_exists(self):
        """Test that patch function is importable."""
        from npuslim.plugins.vllm.model_executor.models.qwen3_moe import (
            patch_qwen3_moe_load_weights,
        )

        assert callable(patch_qwen3_moe_load_weights)

    def test_moe_weight_collector_import(self):
        """Test that MoEWeightCollector can be imported from patch module."""
        from npuslim.plugins.vllm.model_executor.models.qwen3_moe import (
            MoEWeightCollector,
        )

        assert MoEWeightCollector is not None

    def test_apply_patch_function_exists(self):
        """Test that apply patch function exists."""
        from npuslim.plugins.vllm.model_executor.models.qwen3_moe import (
            apply_qwen3_moe_patch,
        )

        assert callable(apply_qwen3_moe_patch)
