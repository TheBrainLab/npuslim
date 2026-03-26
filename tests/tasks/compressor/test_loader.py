# tests/tasks/compressor/test_loader.py
import torch

from npuslim.tasks.compressor.loader import ChunkLoader


def _seed_loader(loader: ChunkLoader, tensor_names: list[str]) -> None:
    loader._resolved_dir = loader.model_path
    loader._weight_map = {name: "model.safetensors" for name in tensor_names}
    loader._tensor_names = list(tensor_names)
    loader._build_layer_tensor_map()
    loader._build_aux_tensor_lists()


def test_chunk_loader_counts_by_layers_not_total_tensors(tmp_path):
    loader = ChunkLoader(model_path=tmp_path, block_name="model.layers", chunk_size=2)
    _seed_loader(
        loader,
        [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "lm_head.weight",
        ],
    )

    assert loader.get_total_tensors() == 6
    assert loader.get_total_layers() == 3
    assert loader.get_chunk_count() == 2


def test_chunk_loader_loads_consecutive_layers_only(tmp_path, monkeypatch):
    loader = ChunkLoader(
        model_path=tmp_path,
        block_name="model.layers",
        chunk_size=2,
        pre_module_names=["model.embed_tokens"],
        post_module_names=["lm_head"],
    )
    _seed_loader(
        loader,
        [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.q_proj.weight",
            "lm_head.weight",
        ],
    )

    monkeypatch.setattr(loader, "_load_tensor", lambda name: torch.tensor([len(name)]))

    chunk0 = loader.load_chunk(0)
    chunk1 = loader.load_chunk(1)

    assert list(chunk0.pre_tensors.keys()) == ["model.embed_tokens.weight"]
    assert list(chunk0.tensors.keys()) == [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]
    assert list(chunk1.tensors.keys()) == [
        "model.layers.2.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    assert chunk0.layer_indices == [0, 1]
    assert chunk1.layer_indices == [2]
    assert list(chunk1.post_tensors.keys()) == ["lm_head.weight"]


def test_chunk_loader_respects_custom_block_name(tmp_path, monkeypatch):
    loader = ChunkLoader(
        model_path=tmp_path,
        block_name="model.decoder.layers",
        chunk_size=2,
        pre_module_names=["model.decoder.embed_tokens"],
        post_module_names=["lm_head"],
    )
    _seed_loader(
        loader,
        [
            "model.decoder.embed_tokens.weight",
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.1.self_attn.q_proj.weight",
            "lm_head.weight",
        ],
    )
    monkeypatch.setattr(loader, "_load_tensor", lambda name: torch.tensor([1.0]))

    chunk0 = loader.load_chunk(0)

    assert loader.get_total_layers() == 2
    assert chunk0.layer_indices == [0, 1]
    assert set(chunk0.tensors.keys()) == {
        "model.decoder.embed_tokens.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.1.self_attn.q_proj.weight",
        "lm_head.weight",
    }


def test_chunk_loader_preserves_pre_post_module_order(tmp_path, monkeypatch):
    loader = ChunkLoader(
        model_path=tmp_path,
        block_name="model.layers",
        chunk_size=2,
        pre_module_names=["model.embed_tokens", "model.rotary_emb"],
        post_module_names=["model.norm", "lm_head"],
    )
    _seed_loader(
        loader,
        [
            "model.rotary_emb.inv_freq",
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "lm_head.weight",
            "model.norm.weight",
        ],
    )
    monkeypatch.setattr(loader, "_load_tensor", lambda name: torch.tensor([1.0]))

    chunk0 = loader.load_chunk(0)

    assert [module.name for module in chunk0.pre_modules] == [
        "model.embed_tokens",
        "model.rotary_emb",
    ]
    assert [module.name for module in chunk0.post_modules] == [
        "model.norm",
        "lm_head",
    ]


def test_chunk_loader_reads_pytorch_bin_index(tmp_path, monkeypatch):
    index_path = tmp_path / "pytorch_model.bin.index.json"
    index_path.write_text(
        """
{
  "metadata": {},
  "weight_map": {
    "model.decoder.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
    "model.decoder.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00002.bin",
    "model.decoder.layers.1.self_attn.q_proj.weight": "pytorch_model-00002-of-00002.bin",
    "lm_head.weight": "pytorch_model-00002-of-00002.bin"
  }
}
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "pytorch_model-00001-of-00002.bin").write_bytes(b"stub")
    (tmp_path / "pytorch_model-00002-of-00002.bin").write_bytes(b"stub")

    loader = ChunkLoader(
        model_path=tmp_path,
        block_name="model.decoder.layers",
        chunk_size=2,
        pre_module_names=["model.decoder.embed_tokens"],
        post_module_names=["lm_head"],
    )

    shard_data = {
        str(tmp_path / "pytorch_model-00001-of-00002.bin"): {
            "model.decoder.embed_tokens.weight": torch.ones(2, 2),
            "model.decoder.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        str(tmp_path / "pytorch_model-00002-of-00002.bin"): {
            "model.decoder.layers.1.self_attn.q_proj.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
        },
    }

    def _fake_torch_load(path, map_location="cpu", weights_only=True):
        return shard_data[str(path)]

    monkeypatch.setattr(torch, "load", _fake_torch_load)

    loader.refresh_index()
    assert loader.get_total_tensors() == 4
    assert loader.get_total_layers() == 2

    chunk = loader.load_chunk(0)
    assert chunk.layer_indices == [0, 1]
    assert "model.decoder.embed_tokens.weight" in chunk.all_tensors()
    assert "model.decoder.layers.0.self_attn.q_proj.weight" in chunk.all_tensors()
    assert "model.decoder.layers.1.self_attn.q_proj.weight" in chunk.all_tensors()
    assert "lm_head.weight" in chunk.all_tensors()
