"""Tests for MoE weight fusion utilities."""
import pytest
import torch


class TestMoEWeightDetection:
    """Test MoE weight detection logic."""

    def test_is_moe_weight_gate_proj(self):
        """Test detection of gate_proj expert weight."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            is_moe_weight,
        )

        assert is_moe_weight("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert is_moe_weight("model.layers.0.mlp.experts.0.gate_proj.weight_scale")

    def test_is_moe_weight_down_proj(self):
        """Test detection of down_proj expert weight."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            is_moe_weight,
        )

        assert is_moe_weight("model.layers.0.mlp.experts.0.down_proj.weight")
        assert is_moe_weight("model.layers.0.mlp.experts.0.down_proj.weight_offset")

    def test_is_not_moe_weight(self):
        """Test that non-expert weights are not detected."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            is_moe_weight,
        )

        assert not is_moe_weight("model.layers.0.mlp.gate.weight")
        assert not is_moe_weight("model.layers.0.self_attn.q_proj.weight")

    def test_get_expert_layer_prefix(self):
        """Test extraction of expert layer prefix."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            get_expert_layer_prefix,
        )

        name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        prefix, expert_id, proj_type = get_expert_layer_prefix(name)
        assert prefix == "model.layers.0.mlp.experts.0"
        assert expert_id == 0
        assert proj_type == "gate_proj"

    def test_get_expert_layer_prefix_up_proj(self):
        """Test extraction for up_proj."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            get_expert_layer_prefix,
        )

        name = "model.layers.5.mlp.experts.127.up_proj.weight_scale"
        prefix, expert_id, proj_type = get_expert_layer_prefix(name)
        assert prefix == "model.layers.5.mlp.experts.127"
        assert expert_id == 127
        assert proj_type == "up_proj"


class TestExpertWeightFusion:
    """Test expert weight fusion logic."""

    @pytest.fixture
    def sample_expert_weights(self):
        """Create sample expert weights for testing."""
        num_experts = 4
        intermediate_size = 128
        hidden_size = 64
        group_size = 32

        weights = {}
        for expert_id in range(num_experts):
            # gate_proj: packed [intermediate//8, hidden]
            weights[f"experts.{expert_id}.gate_proj.weight"] = torch.randint(
                0, 16, (intermediate_size // 8, hidden_size), dtype=torch.int32
            )
            weights[f"experts.{expert_id}.gate_proj.weight_scale"] = torch.rand(
                intermediate_size, hidden_size // group_size
            ).to(torch.bfloat16)
            weights[f"experts.{expert_id}.gate_proj.weight_offset"] = torch.zeros(
                intermediate_size, hidden_size // group_size, dtype=torch.bfloat16
            )
            # up_proj
            weights[f"experts.{expert_id}.up_proj.weight"] = torch.randint(
                0, 16, (intermediate_size // 8, hidden_size), dtype=torch.int32
            )
            weights[f"experts.{expert_id}.up_proj.weight_scale"] = torch.rand(
                intermediate_size, hidden_size // group_size
            ).to(torch.bfloat16)
            weights[f"experts.{expert_id}.up_proj.weight_offset"] = torch.zeros(
                intermediate_size, hidden_size // group_size, dtype=torch.bfloat16
            )
            # down_proj: [hidden//8, intermediate]
            weights[f"experts.{expert_id}.down_proj.weight"] = torch.randint(
                0, 16, (hidden_size // 8, intermediate_size), dtype=torch.int32
            )
            weights[f"experts.{expert_id}.down_proj.weight_scale"] = torch.rand(
                hidden_size, intermediate_size // group_size
            ).to(torch.bfloat16)
            weights[f"experts.{expert_id}.down_proj.weight_offset"] = torch.zeros(
                hidden_size, intermediate_size // group_size, dtype=torch.bfloat16
            )

        return {
            "weights": weights,
            "num_experts": num_experts,
            "intermediate_size": intermediate_size,
            "hidden_size": hidden_size,
            "group_size": group_size,
        }

    def test_fuse_expert_weights_basic(self, sample_expert_weights):
        """Test basic weight fusion."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            fuse_expert_weights,
        )

        result = fuse_expert_weights(
            sample_expert_weights["weights"],
            num_experts=sample_expert_weights["num_experts"],
            intermediate_size=sample_expert_weights["intermediate_size"],
            hidden_size=sample_expert_weights["hidden_size"],
            group_size=sample_expert_weights["group_size"],
        )

        # Check output keys
        assert "w13_weight_packed" in result
        assert "w2_weight_packed" in result
        assert "w13_weight_scale" in result
        assert "w2_weight_scale" in result

        # w13: [num_experts, 2*intermediate, hidden//8]
        expected_w13_shape = (
            sample_expert_weights["num_experts"],
            2 * sample_expert_weights["intermediate_size"],
            sample_expert_weights["hidden_size"] // 8,
        )
        assert result["w13_weight_packed"].shape == expected_w13_shape

        # w2: [num_experts, hidden, intermediate//8]
        expected_w2_shape = (
            sample_expert_weights["num_experts"],
            sample_expert_weights["hidden_size"],
            sample_expert_weights["intermediate_size"] // 8,
        )
        assert result["w2_weight_packed"].shape == expected_w2_shape

    def test_fuse_expert_weights_scale_shapes(self, sample_expert_weights):
        """Test scale/offset shapes after fusion."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            fuse_expert_weights,
        )

        result = fuse_expert_weights(
            sample_expert_weights["weights"],
            num_experts=sample_expert_weights["num_experts"],
            intermediate_size=sample_expert_weights["intermediate_size"],
            hidden_size=sample_expert_weights["hidden_size"],
            group_size=sample_expert_weights["group_size"],
        )

        # w13_scale: [num_experts, 2*intermediate, hidden//group_size]
        expected_w13_scale_shape = (
            sample_expert_weights["num_experts"],
            2 * sample_expert_weights["intermediate_size"],
            sample_expert_weights["hidden_size"] // sample_expert_weights["group_size"],
        )
        assert result["w13_weight_scale"].shape == expected_w13_scale_shape

        # w2_scale: [num_experts, hidden, intermediate//group_size]
        expected_w2_scale_shape = (
            sample_expert_weights["num_experts"],
            sample_expert_weights["hidden_size"],
            sample_expert_weights["intermediate_size"] // sample_expert_weights["group_size"],
        )
        assert result["w2_weight_scale"].shape == expected_w2_scale_shape


class TestMoEWeightCollector:
    """Test MoEWeightCollector integration."""

    def test_weight_collector_collects_expert_weights(self):
        """Test that weight collector collects expert weights."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            MoEWeightCollector,
        )

        collector = MoEWeightCollector(
            layer_prefix="model.layers.0.mlp",
            num_experts=2,
            intermediate_size=128,
            hidden_size=64,
            group_size=32,
        )

        # Add all weights for expert 0
        collector.add_weight(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            torch.zeros(16, 64, dtype=torch.int32),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
            torch.zeros(128, 2, dtype=torch.bfloat16),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.gate_proj.weight_offset",
            torch.zeros(128, 2, dtype=torch.bfloat16),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            torch.zeros(16, 64, dtype=torch.int32),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.up_proj.weight_scale",
            torch.zeros(128, 2, dtype=torch.bfloat16),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.up_proj.weight_offset",
            torch.zeros(128, 2, dtype=torch.bfloat16),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            torch.zeros(8, 128, dtype=torch.int32),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.down_proj.weight_scale",
            torch.zeros(64, 4, dtype=torch.bfloat16),
        )
        collector.add_weight(
            "model.layers.0.mlp.experts.0.down_proj.weight_offset",
            torch.zeros(64, 4, dtype=torch.bfloat16),
        )

        assert collector.has_all_weights(0)
        assert not collector.has_all_weights(1)  # Expert 1 not loaded yet

    def test_weight_collector_is_complete(self):
        """Test is_complete checks all experts."""
        from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
            MoEWeightCollector,
        )

        collector = MoEWeightCollector(
            layer_prefix="model.layers.0.mlp",
            num_experts=2,
            intermediate_size=128,
            hidden_size=64,
            group_size=32,
        )

        # Add weights for both experts
        for expert_id in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert_id}"
            collector.add_weight(f"{prefix}.gate_proj.weight", torch.zeros(16, 64, dtype=torch.int32))
            collector.add_weight(f"{prefix}.gate_proj.weight_scale", torch.zeros(128, 2, dtype=torch.bfloat16))
            collector.add_weight(f"{prefix}.gate_proj.weight_offset", torch.zeros(128, 2, dtype=torch.bfloat16))
            collector.add_weight(f"{prefix}.up_proj.weight", torch.zeros(16, 64, dtype=torch.int32))
            collector.add_weight(f"{prefix}.up_proj.weight_scale", torch.zeros(128, 2, dtype=torch.bfloat16))
            collector.add_weight(f"{prefix}.up_proj.weight_offset", torch.zeros(128, 2, dtype=torch.bfloat16))
            collector.add_weight(f"{prefix}.down_proj.weight", torch.zeros(8, 128, dtype=torch.int32))
            collector.add_weight(f"{prefix}.down_proj.weight_scale", torch.zeros(64, 4, dtype=torch.bfloat16))
            collector.add_weight(f"{prefix}.down_proj.weight_offset", torch.zeros(64, 4, dtype=torch.bfloat16))

        assert collector.is_complete()
