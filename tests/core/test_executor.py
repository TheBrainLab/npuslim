# tests/v2/test_executor.py
"""Tests for SlimEngineV2 and PipelineExecutor."""
import pytest
from unittest.mock import Mock

from npuslim.core.engine import EngineConfig
from npuslim.core.executor import PipelineExecutor
from npuslim.core.config import Config, ExecutionMode
from npuslim.core.algorithm import BaseAlgorithm, step
from npuslim.core.context import AlgorithmContext


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

    algo_config = Config()
    algo = SimpleTestAlgorithm(config=algo_config)
    ctx = AlgorithmContext(config=algo_config, model=mock_model)

    executor = PipelineExecutor(algorithm=algo, context=ctx)
    assert executor.algorithm is algo
    assert executor.context is ctx
