# tests/tasks/compressor/test_task.py
from unittest.mock import MagicMock, patch
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
