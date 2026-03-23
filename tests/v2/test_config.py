# tests/v2/test_config.py
import pytest
from npuslim.v2.config import V2Config, ExecutionMode, ChunkConfig, StreamingConfig


def test_execution_mode_enum():
    """Test ExecutionMode enum values."""
    assert ExecutionMode.FULL.value == "full"
    assert ExecutionMode.LAYER_WISE.value == "layer_wise"
    assert ExecutionMode.CHUNK_WISE.value == "chunk_wise"
    assert ExecutionMode.STREAMING.value == "streaming"


def test_chunk_config_defaults():
    """Test ChunkConfig default values."""
    config = ChunkConfig()
    assert config.size == 1
    assert config.offload_strategy == "lazy"
    assert config.preload_layers is None


def test_streaming_config_defaults():
    """Test StreamingConfig default values."""
    config = StreamingConfig()
    assert config.enabled is True
    assert config.shard_size == "5GB"
    assert config.size_threshold == 4 * 1024 * 1024 * 1024


def test_v2config_validation():
    """Test V2Config validation logic."""
    # CHUNK_WISE mode without chunk config should raise
    with pytest.raises(ValueError, match="chunk config required"):
        V2Config(execution_mode=ExecutionMode.CHUNK_WISE)

    # CHUNK_WISE mode with chunk config should work
    config = V2Config(
        execution_mode=ExecutionMode.CHUNK_WISE,
        chunk=ChunkConfig(size=4)
    )
    assert config.execution_mode == ExecutionMode.CHUNK_WISE
