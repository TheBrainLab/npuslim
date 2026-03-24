"""Tests for distributed execution support."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from npuslim.core.distributed import DistributedManager
from npuslim.core.config import DistributedConfig, DistributedBackend


def test_distributed_config_defaults():
    """Test DistributedConfig default values."""
    config = DistributedConfig()
    assert config.backend == DistributedBackend.NONE
    assert config.world_size == 1
    assert config.rank == 0
    assert config.local_rank == 0
    assert config.tensor_parallel_size == 1
    assert config.pipeline_parallel_size == 1


def test_distributed_manager_none_backend():
    """Test DistributedManager with NONE backend (single process)."""
    config = DistributedConfig(backend=DistributedBackend.NONE)
    manager = DistributedManager(config)

    assert manager.is_distributed is False
    assert manager.is_main_process is True
    assert manager.world_size == 1
    assert manager.rank == 0


def test_distributed_manager_barrier_noop():
    """Test barrier is a no-op with NONE backend."""
    config = DistributedConfig(backend=DistributedBackend.NONE)
    manager = DistributedManager(config)

    # Should not raise any error
    manager.barrier()


def test_distributed_manager_main_process_first():
    """Test main_process_first context manager."""
    config = DistributedConfig(backend=DistributedBackend.NONE)
    manager = DistributedManager(config)

    executed = []
    with manager.main_process_first():
        executed.append("inside")

    assert executed == ["inside"]


def test_distributed_manager_prepare_model_noop():
    """Test prepare_model returns unchanged with NONE backend."""
    config = DistributedConfig(backend=DistributedBackend.NONE)
    manager = DistributedManager(config)

    model = Mock()
    optimizer = Mock()

    prepared_model, prepared_optimizer, _, _ = manager.prepare_model(
        model, optimizer, None, None
    )

    assert prepared_model is model
    assert prepared_optimizer is optimizer


def test_distributed_config_accelerate():
    """Test DistributedConfig with accelerate backend."""
    config = DistributedConfig(
        backend=DistributedBackend.ACCELERATE,
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
    )
    assert config.backend == DistributedBackend.ACCELERATE
    assert config.mixed_precision == "fp16"
    assert config.gradient_accumulation_steps == 4


def test_distributed_config_torch_distributed():
    """Test DistributedConfig with torch.distributed backend."""
    config = DistributedConfig(
        backend=DistributedBackend.TORCH_DISTRIBUTED,
        world_size=4,
        rank=1,
        local_rank=1,
    )
    assert config.backend == DistributedBackend.TORCH_DISTRIBUTED
    assert config.world_size == 4
    assert config.rank == 1


def test_distributed_config_deepspeed():
    """Test DistributedConfig with deepspeed backend."""
    config = DistributedConfig(
        backend=DistributedBackend.DEEPSPEED,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    assert config.backend == DistributedBackend.DEEPSPEED
    assert config.tensor_parallel_size == 2
    assert config.pipeline_parallel_size == 2
