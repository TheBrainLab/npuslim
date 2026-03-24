from unittest.mock import Mock

from npuslim.algorithms.base import BaseAlgorithm
from npuslim.config.schema import Config, ExecutionMode, StreamingConfig
from npuslim.core.context import AlgorithmContext
from npuslim.core.executor import PipelineExecutor


class NoopAlgorithm(BaseAlgorithm):
    execution_mode = ExecutionMode.STREAMING
    chunk_size = 2


def test_executor_uses_model_chunk_lifecycle_in_streaming_mode():
    model = Mock()
    model.get_total_layers.return_value = 5
    model.load_chunk.side_effect = [
        [{"name": "model.layers.0"}, {"name": "model.layers.1"}],
        [{"name": "model.layers.2"}, {"name": "model.layers.3"}],
        [{"name": "model.layers.4"}],
    ]

    cfg = Config(
        execution_mode=ExecutionMode.STREAMING,
        streaming=StreamingConfig(enabled=True),
    )
    ctx = AlgorithmContext(config=cfg, model=model)
    algo = NoopAlgorithm(config=cfg)

    executor = PipelineExecutor(algorithm=algo, context=ctx)
    executor.run()

    assert model.load_chunk.call_count == 3
    assert model.release_chunk.call_count == 3


def test_executor_finalize_calls_stream_saver_once():
    model = Mock()
    model.get_total_layers.return_value = 2
    model.load_chunk.return_value = [{"name": "model.layers.0"}, {"name": "model.layers.1"}]

    cfg = Config(
        execution_mode=ExecutionMode.STREAMING,
        streaming=StreamingConfig(enabled=True),
    )
    ctx = AlgorithmContext(config=cfg, model=model)
    ctx._stream_saver = Mock()
    algo = NoopAlgorithm(config=cfg)

    executor = PipelineExecutor(algorithm=algo, context=ctx)
    executor.run()

    ctx._stream_saver.finalize.assert_called_once()
