"""Shared fixtures for config tests."""
import pytest


@pytest.fixture
def sample_config_dict():
    """Sample config dictionary for testing."""
    return {
        "metadata": {"name": "Test", "description": "Test config"},
        "resources": [
            {"id": "model1", "type": "TestModel", "path": "/model"},
            {"id": "data1", "type": "TestDataset", "num_samples": 128}
        ],
        "recipe": [
            {
                "name": "Task1",
                "type": "compressor",
                "model": "@model1",
                "data": "@data1",
                "algorithm": {"type": "GPTQ", "wbits": 4}
            }
        ]
    }


@pytest.fixture
def speculative_config_dict():
    """Sample speculative decoding config."""
    return {
        "metadata": {"name": "Speculative"},
        "resources": [
            {"id": "main", "type": "Qwen3Model", "path": "Qwen/Qwen3-32B"},
            {"id": "draft", "type": "Qwen3Model", "path": "Qwen/Qwen3-0.6B"}
        ],
        "recipe": [
            {
                "name": "Spec",
                "type": "speculative",
                "main_model": "@main",
                "draft_model": "@draft"
            }
        ]
    }
