import json
from pathlib import Path
from types import SimpleNamespace

import torch

from npuslim.savers.hf_saver import StreamingHuggingFaceSaver


def test_hf_saver_streaming_writes_shards_and_index(tmp_path):
    output_dir = tmp_path / "out"
    saver = StreamingHuggingFaceSaver(output_dir=output_dir, size_threshold=16)

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
    (source_dir / "tf_model.h5").write_text("do-not-copy", encoding="utf-8")
    (source_dir / "custom_code.py").write_text("x=1\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    saver = StreamingHuggingFaceSaver(output_dir=output_dir, size_threshold=1024)
    saver.set_source(source_dir, model_hub="hf", model_kwargs={})
    saver.finalize()

    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "custom_code.py").exists()
    assert not (output_dir / "model.safetensors").exists()
    assert not (output_dir / "tf_model.h5").exists()


def test_hf_saver_writes_ascend_quant_description(tmp_path):
    output_dir = tmp_path / "out"
    saver = StreamingHuggingFaceSaver(output_dir=output_dir, size_threshold=1024)

    model_config = SimpleNamespace(
        ascend_quant_config={
            "model_quant_type": "W8A8_DYNAMIC",
            "group_size": -1,
            "include_g_idx": False,
            "has_offset": False,
        }
    )
    saver.set_hf_assets(model_config=model_config)
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(2, 1),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
    }
    tensor_types = {
        "model.embed_tokens.weight": "FLOAT",
        "model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.self_attn.q_proj.weight_scale": "W8A8_DYNAMIC",
        "model.layers.0.self_attn.q_norm.weight": "FLOAT",
    }
    saver.add_tensors(tensors, tensor_types=tensor_types)
    saver.finalize()

    desc_path = output_dir / "quant_model_description.json"
    assert desc_path.exists()
    with open(desc_path, "r", encoding="utf-8") as f:
        description = json.load(f)

    assert description["model_quant_type"] == "W8A8_DYNAMIC"
    assert description["model.embed_tokens.weight"] == "FLOAT"
    assert description["model.layers.0.self_attn.q_norm.weight"] == "FLOAT"
    assert description["model.layers.0.self_attn.q_proj.weight"] == "W8A8_DYNAMIC"
    assert (
        description["model.layers.0.self_attn.q_proj.weight_scale"] == "W8A8_DYNAMIC"
    )
