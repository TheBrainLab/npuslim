import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import json
import torch
from safetensors.torch import save_file

from npuslim.streaming import StreamSaver
from npuslim.streaming.streaming import SafeTensorIndex, ShardTensorReader


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


def test_safe_tensor_index_parses_weight_map(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1234},
                "weight_map": {
                    "model.layers.0.weight": "model-00001.safetensors",
                    "model.layers.1.weight": "model-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    index = SafeTensorIndex(index_path)

    assert index.total_size == 1234
    assert index.shard_for("model.layers.0.weight") == "model-00001.safetensors"


def test_shard_tensor_reader_reads_tensor_by_name(tmp_path):
    shard_path = tmp_path / "model-00001.safetensors"
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    save_file({"model.layers.0.weight": tensor}, shard_path)

    reader = ShardTensorReader(tmp_path)
    out = reader.get_tensor("model-00001.safetensors", "model.layers.0.weight")

    assert torch.equal(out, tensor)


def test_shard_tensor_reader_caches_opened_shards(tmp_path):
    shard_path = tmp_path / "model-00001.safetensors"
    tensor = torch.ones(2, 2)
    save_file({"model.layers.0.weight": tensor}, shard_path)

    reader = ShardTensorReader(tmp_path)
    reader.get_tensor("model-00001.safetensors", "model.layers.0.weight")
    reader.get_tensor("model-00001.safetensors", "model.layers.0.weight")

    assert "model-00001.safetensors" in reader.opened_shards
