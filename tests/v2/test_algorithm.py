# tests/v2/test_algorithm.py
"""Tests for BaseAlgorithm and @step decorator."""
import pytest
from npuslim.v2.algorithm import BaseAlgorithm, step, StepInfo
from npuslim.v2.config import V2Config, ExecutionMode


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
    config = V2Config()
    algo = MockAlgorithm(config=config)
    steps = algo.get_steps()

    assert len(steps) == 2
    assert steps[0].order == 1
    assert steps[1].order == 2


def test_algorithm_lifecycle_hooks():
    """Test algorithm has default lifecycle hooks."""
    config = V2Config()
    algo = MockAlgorithm(config=config)

    # These should be callable without error
    algo.on_start(None)
    algo.on_chunk_enter(None)
    algo.on_chunk_exit(None)
    algo.on_finish(None)
