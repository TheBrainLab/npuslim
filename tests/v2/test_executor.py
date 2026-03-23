# tests/v2/test_executor.py
"""Tests for SlimEngineV2 and PipelineExecutor."""
import pytest
from unittest.mock import Mock, MagicMock

from npuslim.v2.engine import SlimEngineV2, EngineConfig
from npuslim.v2.executor import PipelineExecutor
from npuslim.v2.config import V2Config, ExecutionMode
from npuslim.v2.algorithm import BaseAlgorithm, step
from npuslim.v2.context import AlgorithmContext


class SimpleTestAlgorithm(BaseAlgorithm):
    """Simple test algorithm."""
    execution_mode = ExecutionMode.FULL

    @step(order=1, requires=["layer"], produces=["output"])
    def process(self, context):
        return {"output": "done"}


def test_engine_config():
    """Test engine configuration."""
    config = EngineConfig(
        model_path="test/model",
        output_dir="test/output",
        pipeline=[{"type": "ptq", "algo_name": "test"}],
    )
    assert config.model_path == "test/model"


def test_pipeline_executor_creation():
    """Test PipelineExecutor can be created."""
    mock_model = Mock()
    mock_model.get_layers.return_value = []

    algo_config = V2Config()
    algo = SimpleTestAlgorithm(config=algo_config)
    ctx = AlgorithmContext(config=algo_config, model=mock_model)

    executor = PipelineExecutor(algorithm=algo, context=ctx)
    assert executor.algorithm is algo
    assert executor.context is ctx
