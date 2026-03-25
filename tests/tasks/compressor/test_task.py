# tests/tasks/compressor/test_task.py
from unittest.mock import MagicMock
from npuslim.tasks.compressor.task import CompressorTask


def test_compressor_task_init():
    task = CompressorTask(
        name="test_compress",
        model="@qwen3",
        algorithm={"type": "INT8Dynamic", "wbits": 8},
        execution={"mode": "streaming", "chunk_size": 2},
        saver={"type": "HuggingFaceSaver", "output_dir": "/tmp/out"},
        resource_manager=MagicMock(),
    )

    assert task.name == "test_compress"
    assert task.chunk_size == 2
    assert task.mode == "streaming"


def test_compressor_task_requires_resource_manager():
    task = CompressorTask(
        name="test",
        model="@qwen3",
        algorithm={"type": "INT8Dynamic"},
    )
    # Without resource_manager, should raise ValueError on execute
    try:
        task.execute()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "resource_manager" in str(e)


def test_create_loader_uses_model_block_name():
    task = CompressorTask(
        name="test_compress",
        model="@qwen3",
        algorithm={"type": "INT8Dynamic", "wbits": 8},
        execution={"mode": "streaming", "chunk_size": 2},
        resource_manager=MagicMock(),
    )
    task._model_obj = MagicMock(
        path_str="/tmp/mock-model",
        model_hub="hf",
        model_kwargs={},
        block_name="model.decoder.layers",
        layers_path="model.decoder.layers",
        pre_transformer_module_names=["model.decoder.embed_tokens"],
        post_transformer_module_names=["model.decoder.final_layer_norm", "lm_head"],
    )

    loader = task._create_loader()

    assert loader.block_name == "model.decoder.layers"
    assert loader.pre_module_names == ["model.decoder.embed_tokens"]
    assert loader.post_module_names == ["model.decoder.final_layer_norm", "lm_head"]
