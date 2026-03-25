# tests/tasks/compressor/test_loader.py
import pytest
from pathlib import Path
from npuslim.tasks.compressor.loader import ChunkLoader


def test_chunk_loader_resolve_local_path(tmp_path):
    """Test resolving local model path."""
    loader = ChunkLoader(model_path=tmp_path, block_name="model.layers")
    assert loader.model_path == tmp_path


def test_chunk_loader_chunk_count(tmp_path):
    """Test chunk count calculation."""
    loader = ChunkLoader(model_path=tmp_path, block_name="model.layers", chunk_size=2)
    # Without index, returns 0
    assert loader.get_chunk_count() == 0


def test_chunk_loader_total_layers(tmp_path):
    """Test total layers defaults to 0."""
    loader = ChunkLoader(model_path=tmp_path, block_name="model.layers")
    assert loader.get_total_layers() == 0
