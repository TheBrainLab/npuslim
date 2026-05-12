from types import SimpleNamespace

import torch

from npuslim.algorithms.quantization.int8_dynamic.int8_dynamic_algo import (
    INT8DynamicAlgorithm,
)
from npuslim.core import AlgorithmRegistry
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


def test_int8_dynamic_algorithm_registry():
    cls = AlgorithmRegistry.get("INT8Dynamic")
    assert cls is INT8DynamicAlgorithm


def test_int8_dynamic_quantizes_packed_moe_expert_params():
    algo = INT8DynamicAlgorithm(wbits=8, w_quant_method="per-channel")
    model_obj = SimpleNamespace(quantized=False)
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    chunk = ChunkContext(
        chunk_index=0,
        layers=[
            LayerInfo(
                name="model.language_model.layers.0",
                index=0,
                tensors={
                    "self_attn.q_proj.weight": torch.randn(16, 16, dtype=torch.float32),
                    "mlp.experts.gate_up_proj": torch.randn(
                        4, 12, 16, dtype=torch.float32
                    ),
                    "mlp.experts.down_proj": torch.randn(4, 16, 8, dtype=torch.float32),
                },
            )
        ],
    )

    algo.on_start()
    out_chunk = algo.process_chunk(chunk)
    algo.on_finish()

    tensors = out_chunk.layers[0].tensors
    assert "self_attn.q_proj.weight_scale" in tensors
    assert "mlp.experts.gate_up_proj_scale" in tensors
    assert "mlp.experts.down_proj_scale" in tensors
    assert tensors["mlp.experts.gate_up_proj"].shape == (4, 12, 16)
    assert tensors["mlp.experts.down_proj"].shape == (4, 16, 8)
    assert tensors["mlp.experts.gate_up_proj_scale"].shape == (4, 12, 1)
    assert tensors["mlp.experts.down_proj_scale"].shape == (4, 16, 1)


def test_int8_dynamic_does_not_quantize_unrelated_3d_tensors():
    algo = INT8DynamicAlgorithm(wbits=8, w_quant_method="per-channel")
    model_obj = SimpleNamespace(quantized=False)
    model_config = SimpleNamespace()
    algo.set_runtime_context(model_obj=model_obj, model_config=model_config)

    chunk = ChunkContext(
        chunk_index=0,
        layers=[
            LayerInfo(
                name="model.layers.0",
                index=0,
                tensors={
                    "self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.float32),
                    "mlp.random_3d": torch.randn(2, 3, 4, dtype=torch.float32),
                },
            )
        ],
    )

    algo.on_start()
    out_chunk = algo.process_chunk(chunk)
    algo.on_finish()

    tensors = out_chunk.layers[0].tensors
    assert "self_attn.q_proj.weight_scale" in tensors
    assert "mlp.random_3d_scale" not in tensors
