"""Tests for config parser."""
import pytest
import tempfile
from pathlib import Path

from npuslim.config.parser import (
    parse_config,
    SlimConfig,
    MetadataConfig,
    ResourceConfig,
    RecipeTaskConfig,
    AlgorithmConfig
)


class TestParseConfig:
    """Tests for parse_config function."""

    def test_parse_from_dict(self, sample_config_dict):
        """Parse config from dictionary."""
        config = parse_config(sample_config_dict)

        assert isinstance(config, SlimConfig)
        assert config.metadata.name == "Test"
        assert len(config.resources) == 2
        assert len(config.recipe) == 1

    def test_parse_from_file(self, sample_config_dict):
        """Parse config from YAML file."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_dict, f)
            f.flush()

            config = parse_config(f.name)

            assert config.metadata.name == "Test"
            assert len(config.resources) == 2

            Path(f.name).unlink()

    def test_parse_metadata(self, sample_config_dict):
        """Parse metadata section."""
        config = parse_config(sample_config_dict)

        assert config.metadata.name == "Test"
        assert config.metadata.description == "Test config"

    def test_parse_resources(self, sample_config_dict):
        """Parse resources with extra fields."""
        config = parse_config(sample_config_dict)

        assert len(config.resources) == 2

        model = config.resources[0]
        assert model.id == "model1"
        assert model.type == "TestModel"
        assert model.extra["path"] == "/model"

        data = config.resources[1]
        assert data.id == "data1"
        assert data.type == "TestDataset"
        assert data.extra["num_samples"] == 128

    def test_parse_recipe_with_algorithm(self, sample_config_dict):
        """Parse recipe task with algorithm config."""
        config = parse_config(sample_config_dict)

        task = config.recipe[0]
        assert task.name == "Task1"
        assert task.type == "compressor"
        assert task.model == "@model1"
        assert task.data == "@data1"

        assert task.algorithm is not None
        assert task.algorithm.type == "GPTQ"
        assert task.algorithm.extra["wbits"] == 4

    def test_parse_speculative_task(self, speculative_config_dict):
        """Parse speculative decoding task."""
        config = parse_config(speculative_config_dict)

        task = config.recipe[0]
        assert task.type == "speculative"
        assert task.main_model == "@main"
        assert task.draft_model == "@draft"

    def test_parse_empty_sections(self):
        """Parse config with empty resources and recipe."""
        config = parse_config({"metadata": {}, "resources": [], "recipe": []})

        assert config.resources == []
        assert config.recipe == []

    def test_parse_algorithm_string(self):
        """Parse algorithm as string (no extra config)."""
        config = parse_config({
            "metadata": {},
            "resources": [{"id": "m", "type": "Model"}],
            "recipe": [{"name": "T", "type": "compressor", "model": "@m", "algorithm": "GPTQ"}]
        })

        assert config.recipe[0].algorithm.type == "GPTQ"
        assert config.recipe[0].algorithm.extra == {}

    def test_parse_recipe_execution_mode(self):
        """Parse recipe execution mode and chunk size."""
        config = parse_config({
            "metadata": {"name": "x"},
            "resources": [{"id": "m", "type": "Qwen3Model", "path": "Qwen/Qwen3-0.6B"}],
            "recipe": [
                {
                    "name": "quant",
                    "type": "QuantizeTask",
                    "model": "@m",
                    "execution": {"mode": "streaming", "chunk_size": 1},
                }
            ],
        })

        assert config.recipe[0].execution.mode == "streaming"
        assert config.recipe[0].execution.chunk_size == 1


class TestSlimConfig:
    """Tests for SlimConfig methods."""

    def test_get_resource_by_id(self, sample_config_dict):
        """Get resource by ID."""
        config = parse_config(sample_config_dict)

        r = config.get_resource_by_id("model1")
        assert r is not None
        assert r.type == "TestModel"

    def test_get_resource_by_id_with_at_prefix(self, sample_config_dict):
        """Get resource by ID with @ prefix."""
        config = parse_config(sample_config_dict)

        r = config.get_resource_by_id("@data1")
        assert r is not None
        assert r.type == "TestDataset"

    def test_get_resource_by_id_not_found(self, sample_config_dict):
        """Get non-existent resource returns None."""
        config = parse_config(sample_config_dict)

        r = config.get_resource_by_id("nonexistent")
        assert r is None

    def test_get_resources_by_type(self, sample_config_dict):
        """Filter resources by type suffix."""
        config = parse_config(sample_config_dict)

        models = config.get_resources_by_type("Model")
        assert len(models) == 1
        assert models[0].id == "model1"
