# tests/v2/test_context.py
import pytest
from unittest.mock import MagicMock, Mock
from npuslim.v2.context import AlgorithmContext
from npuslim.v2.config import V2Config, ExecutionMode, StreamingConfig


def test_context_creation():
    """Test basic context creation."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    config = V2Config()
    ctx = AlgorithmContext(config=config, model=mock_model)

    assert ctx.config is config
    assert ctx.model is mock_model
    assert ctx.dataloader is None
    assert ctx.current_chunk is None


def test_context_streaming_properties():
    """Test streaming-related properties."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    # Without streaming config
    ctx = AlgorithmContext(config=V2Config(), model=mock_model)
    assert ctx.is_streaming is False

    # With streaming config
    streaming_cfg = StreamingConfig(enabled=True)
    ctx = AlgorithmContext(
        config=V2Config(streaming=streaming_cfg),
        model=mock_model
    )
    assert ctx.is_streaming is True


def test_context_chunk_management():
    """Test chunk management methods."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=V2Config(), model=mock_model)

    # Set chunk
    chunk = {"layers": [Mock(), Mock()], "index": 0}
    ctx.set_current_chunk(chunk)

    assert ctx.current_chunk == chunk
    assert ctx.layer_index == 0

    # Advance layer
    ctx.advance_layer()
    assert ctx.layer_index == 1


def test_context_intermediates():
    """Test intermediate storage."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=V2Config(), model=mock_model)

    ctx.set_intermediate("hessian", Mock())
    assert ctx.get_intermediate("hessian") is not None
    assert ctx.get_intermediate("nonexistent") is None

    ctx.clear_intermediates()
    assert ctx.get_intermediate("hessian") is None
