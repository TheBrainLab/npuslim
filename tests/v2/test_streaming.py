# tests/v2/test_streaming.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch

from npuslim.v2.streaming import StreamSaver
from npuslim.v2.config import ChunkConfig


def test_stream_saver_add_tensor():
    """Test adding tensors to buffer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = StreamSaver(output_dir=Path(tmpdir), size_threshold=1024*1024)

        tensor = torch.randn(10, 10)
        saver.add_tensor("layer.0.weight", tensor)

        assert len(saver.buffer) == 1
        assert saver.buffer_size > 0


def test_stream_saver_flush():
    """Test flushing buffer to safetensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = StreamSaver(output_dir=Path(tmpdir))

        tensor = torch.randn(100, 100)
        saver.add_tensor("test.weight", tensor)
        shard_name = saver.flush()

        assert shard_name is not None
        assert (Path(tmpdir) / shard_name).exists()


def test_stream_saver_auto_flush():
    """Test auto-flush when threshold exceeded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = StreamSaver(output_dir=Path(tmpdir), size_threshold=1000)

        # First tensor fits
        saver.add_tensor("layer.0.weight", torch.randn(10, 10))
        assert len(saver.buffer) == 1

        # Second tensor triggers flush
        saver.add_tensor("layer.1.weight", torch.randn(100, 100))
        assert saver.shard_counter >= 1


def test_stream_saver_finalize():
    """Test finalizing creates index file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = StreamSaver(output_dir=Path(tmpdir))

        saver.add_tensor("layer.0.weight", torch.randn(10, 10))

        mock_config = Mock()
        mock_config.save_pretrained = Mock()

        saver.finalize(model_config=mock_config)

        assert (Path(tmpdir) / "model.safetensors.index.json").exists()
