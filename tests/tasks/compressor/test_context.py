# tests/tasks/compressor/test_context.py
import torch
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


def test_layer_info_creation():
    layer = LayerInfo(
        name="model.layers.0",
        index=0,
        tensors={"self_attn.q_proj.weight": torch.randn(64, 64)},
    )
    assert layer.name == "model.layers.0"
    assert layer.index == 0
    assert "self_attn.q_proj.weight" in layer.tensors


def test_chunk_context_all_tensors():
    chunk = ChunkContext(
        chunk_index=0,
        layers=[
            LayerInfo(name="layers.0", index=0, tensors={"w1": torch.tensor(1.0)}),
            LayerInfo(name="layers.1", index=1, tensors={"w2": torch.tensor(2.0)}),
        ],
    )
    all_tensors = chunk.all_tensors()
    assert "layers.0.w1" in all_tensors
    assert "layers.1.w2" in all_tensors
    assert all_tensors["layers.0.w1"].item() == 1.0
