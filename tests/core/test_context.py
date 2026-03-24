# tests/v2/test_context.py
import pytest
from unittest.mock import MagicMock, Mock
from npuslim.core.context import AlgorithmContext
from npuslim.core.config import Config, ExecutionMode, StreamingConfig


def test_context_creation():
    """Test basic context creation."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    config = Config()
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
    ctx = AlgorithmContext(config=Config(), model=mock_model)
    assert ctx.is_streaming is False

    # With streaming config
    streaming_cfg = StreamingConfig(enabled=True)
    ctx = AlgorithmContext(
        config=Config(streaming=streaming_cfg),
        model=mock_model
    )
    assert ctx.is_streaming is True


def test_context_chunk_management():
    """Test chunk management methods."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)

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

    ctx = AlgorithmContext(config=Config(), model=mock_model)

    ctx.set_intermediate("hessian", Mock())
    assert ctx.get_intermediate("hessian") is not None
    assert ctx.get_intermediate("nonexistent") is None

    ctx.clear_intermediates()
    assert ctx.get_intermediate("hessian") is None


def test_context_emit_with_stream_saver():
    """Test emit() method with a mock StreamSaver."""
    import torch

    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)

    # Mock StreamSaver
    mock_saver = Mock()
    mock_saver.add_tensor = Mock()
    ctx._stream_saver = mock_saver

    # Test emit
    tensor = torch.randn(2, 2)
    ctx.emit("weight", tensor)

    mock_saver.add_tensor.assert_called_once_with("weight", tensor)


def test_context_flush_with_stream_saver():
    """Test flush() method with a mock StreamSaver."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)

    # Mock StreamSaver
    mock_saver = Mock()
    mock_saver.flush = Mock(return_value="/path/to/output")
    ctx._stream_saver = mock_saver

    # Test flush
    result = ctx.flush()

    assert result == "/path/to/output"
    mock_saver.flush.assert_called_once()


def test_context_emit_raises_without_stream_saver():
    """Test emit() raises RuntimeError when StreamSaver not initialized."""
    import torch

    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)
    # _stream_saver is None by default

    tensor = torch.randn(2, 2)
    with pytest.raises(RuntimeError, match="StreamSaver not initialized"):
        ctx.emit("weight", tensor)


def test_context_flush_returns_none_without_stream_saver():
    """Test flush() returns None when StreamSaver not initialized."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)
    # _stream_saver is None by default

    result = ctx.flush()
    assert result is None


def test_context_current_layer_property():
    """Test current_layer property."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)

    # No chunk set
    assert ctx.current_layer is None

    # Chunk with layers
    layer1 = Mock()
    layer2 = Mock()
    chunk = {"layers": [layer1, layer2], "index": 0}
    ctx.set_current_chunk(chunk)

    assert ctx.current_layer is layer1

    ctx.advance_layer()
    assert ctx.current_layer is layer2

    # Beyond range
    ctx.advance_layer()
    assert ctx.current_layer is None


def test_context_current_layer_name_property():
    """Test current_layer_name property."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    ctx = AlgorithmContext(config=Config(), model=mock_model)

    # No chunk set
    assert ctx.current_layer_name is None

    # Layer with dict-style access
    chunk = {"layers": [{"name": "layer.0"}, {"name": "layer.1"}], "index": 0}
    ctx.set_current_chunk(chunk)

    assert ctx.current_layer_name == "layer.0"

    ctx.advance_layer()
    assert ctx.current_layer_name == "layer.1"

    # Layer with attribute-style access
    layer_with_attr = Mock()
    layer_with_attr.name = "layer.2"
    chunk2 = {"layers": [layer_with_attr], "index": 1}
    ctx.set_current_chunk(chunk2)

    assert ctx.current_layer_name == "layer.2"

    # Layer without name attribute
    layer_no_name = Mock(spec=[])  # No attributes
    chunk3 = {"layers": [layer_no_name], "index": 2}
    ctx.set_current_chunk(chunk3)

    assert ctx.current_layer_name == "unknown"
