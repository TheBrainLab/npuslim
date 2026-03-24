# tests/v2/test_algorithm.py
"""Tests for BaseAlgorithm and @step decorator."""
import pytest
from unittest.mock import MagicMock

from npuslim.core.algorithm import BaseAlgorithm, step, StepInfo
from npuslim.core.config import Config, ExecutionMode
from npuslim.core.context import AlgorithmContext
from npuslim.core.step_executor import StepExecutor


class MockAlgorithm(BaseAlgorithm):
    """Mock algorithm for testing."""
    execution_mode = ExecutionMode.CHUNK_WISE
    chunk_size = 4

    @step(order=1, requires=["layer", "calib_data"], produces=["hessian"])
    def observe(self, context) -> dict:
        return {"hessian": "mock_hessian"}

    @step(order=2, requires=["hessian"], produces=["quantized"])
    def quantize(self, context) -> dict:
        return {"quantized": "mock_quantized"}


def test_step_decorator():
    """Test @step decorator attaches info to method."""
    assert hasattr(MockAlgorithm.observe, "_step_info")
    info = MockAlgorithm.observe._step_info
    assert info.order == 1
    assert info.requires == ["layer", "calib_data"]
    assert info.produces == ["hessian"]


def test_algorithm_collects_steps():
    """Test algorithm collects steps from decorated methods."""
    config = Config()
    algo = MockAlgorithm(config=config)
    steps = algo.get_steps()

    assert len(steps) == 2
    assert steps[0].order == 1
    assert steps[1].order == 2


def test_algorithm_lifecycle_hooks():
    """Test algorithm has default lifecycle hooks."""
    config = Config()
    algo = MockAlgorithm(config=config)

    # These should be callable without error
    algo.on_start(None)
    algo.on_chunk_enter(None)
    algo.on_chunk_exit(None)
    algo.on_finish(None)


def test_step_executor_gathers_inputs_and_executes():
    """Test StepExecutor gathers inputs and executes steps in order."""
    # Setup mock context
    mock_model = MagicMock()
    mock_model.get_layers.return_value = []
    config = Config()
    context = AlgorithmContext(config=config, model=mock_model)

    # Track execution order
    execution_order = []

    # Create steps as standalone functions (StepExecutor calls method directly)
    @step(order=1, produces=["result_a"])
    def step_a(ctx) -> dict:
        execution_order.append("a")
        return {"result_a": "value_a"}

    @step(order=2, requires=["result_a"], produces=["result_b"])
    def step_b(ctx, result_a: str) -> dict:
        execution_order.append("b")
        # Verify intermediate was passed
        assert result_a == "value_a"
        return {"result_b": "value_b"}

    # Get steps directly from decorated functions
    steps = [step_a._step_info, step_b._step_info]

    executor = StepExecutor(context, steps)
    result = executor.execute()

    # Verify steps executed in order
    assert execution_order == ["a", "b"]
    # Verify intermediates were stored
    assert "result_a" in executor.intermediates
    assert executor.intermediates["result_a"] == "value_a"
