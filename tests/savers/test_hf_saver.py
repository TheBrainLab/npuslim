# tests/savers/test_hf_saver.py
import torch
from pathlib import Path
from npuslim.savers.hf_saver import HuggingFaceSaver


def test_hf_saver_add_and_flush(tmp_path):
    saver = HuggingFaceSaver(output_dir=tmp_path, size_threshold=1024*1024)

    saver.add_tensor("layer.0.weight", torch.randn(10, 10))
    saver.add_tensor("layer.0.bias", torch.randn(10))

    # Should flush on finalize
    saver.finalize()

    # Check output
    index_file = tmp_path / "model.safetensors.index.json"
    assert index_file.exists()


def test_hf_saver_auto_flush(tmp_path):
    saver = HuggingFaceSaver(output_dir=tmp_path, size_threshold=100)

    # Small tensors should auto-flush when threshold exceeded
    saver.add_tensor("w1", torch.randn(10, 10))  # 400 bytes
    saver.add_tensor("w2", torch.randn(10, 10))  # 400 bytes

    saver.finalize()
    assert (tmp_path / "model.safetensors.index.json").exists()
