import json
from pathlib import Path

import torch

from npuslim.savers.hf_saver import HuggingFaceSaver


def test_hf_saver_streaming_writes_shards_and_index(tmp_path):
    output_dir = tmp_path / "out"
    saver = HuggingFaceSaver(output_dir=output_dir, size_threshold=16)

    saver.add_tensor("model.layers.0.a.weight", torch.ones(4, dtype=torch.float32))
    saver.add_tensor("model.layers.0.b.weight", torch.ones(4, dtype=torch.float32))
    saver.finalize()

    index_path = output_dir / "model.safetensors.index.json"
    assert index_path.exists()

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    assert set(weight_map.keys()) == {
        "model.layers.0.a.weight",
        "model.layers.0.b.weight",
    }
    assert len(set(weight_map.values())) == 2
    assert index["metadata"]["total_size"] > 0


def test_hf_saver_copies_non_weight_aux_files_from_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source_dir / "model.safetensors").write_text("do-not-copy", encoding="utf-8")
    (source_dir / "custom_code.py").write_text("x=1\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    saver = HuggingFaceSaver(output_dir=output_dir, size_threshold=1024)
    saver.set_source(source_dir, model_hub="hf", model_kwargs={})
    saver.finalize()

    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "custom_code.py").exists()
    assert not (output_dir / "model.safetensors").exists()
