"""Distributed execution support for NPUSlim v2.

This module provides distributed training/quantization support using
various backends (accelerate, torch.distributed, deepspeed).
"""
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from loguru import logger

from npuslim.v2.config import DistributedConfig, DistributedBackend


class DistributedManager:
    """
    Manages distributed execution context.

    Provides a unified interface for distributed operations regardless
    of the underlying backend (accelerate, torch.distributed, deepspeed).
    """

    def __init__(self, config: DistributedConfig):
        self.config = config
        self._accelerator = None
        self._model = None
        self._optimizer = None
        self._setup()

    def _setup(self) -> None:
        """Initialize the distributed backend."""
        if self.config.backend == DistributedBackend.NONE:
            logger.info("Running in single-process mode")
            return

        if self.config.backend == DistributedBackend.ACCELERATE:
            self._setup_accelerate()
        elif self.config.backend == DistributedBackend.TORCH_DISTRIBUTED:
            self._setup_torch_distributed()
        elif self.config.backend == DistributedBackend.DEEPSPEED:
            self._setup_deepspeed()

    def _setup_accelerate(self) -> None:
        """Initialize HuggingFace Accelerate backend."""
        try:
            from accelerate import Accelerator
            from accelerate.utils import DistributedType
        except ImportError:
            raise ImportError(
                "accelerate is required for ACCELERATE backend. "
                "Install with: pip install accelerate"
            )

        self._accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        logger.info(
            f"Accelerate initialized: "
            f"rank={self._accelerator.process_index}, "
            f"mixed_precision={self.config.mixed_precision}"
        )

    def _setup_torch_distributed(self) -> None:
        """Initialize native torch.distributed backend."""
        import torch
        import torch.distributed as dist
        import os

        if not dist.is_initialized():
            # Get info from environment variables (set by torchrun)
            rank = int(os.environ.get("RANK", self.config.rank))
            world_size = int(os.environ.get("WORLD_SIZE", self.config.world_size))
            local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))

            dist.init_process_group(
                backend=self.config.backend_init_method,
                rank=rank,
                world_size=world_size,
            )

            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            logger.info(
                f"torch.distributed initialized: "
                f"rank={rank}, world_size={world_size}, local_rank={local_rank}"
            )

    def _setup_deepspeed(self) -> None:
        """Initialize DeepSpeed backend."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "deepspeed is required for DEEPSPEED backend. "
                "Install with: pip install deepspeed"
            )

        # DeepSpeed is typically initialized via deepspeed.initialize()
        # which wraps model and optimizer. We'll handle this in prepare_model().
        logger.info("DeepSpeed backend selected - will initialize during model preparation")

    # === Properties ===

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.config.backend != DistributedBackend.NONE

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if self.config.backend == DistributedBackend.NONE:
            return True

        if self._accelerator is not None:
            return self._accelerator.is_main_process

        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0

        return self.config.rank == 0

    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        if self.config.backend == DistributedBackend.NONE:
            return 1

        if self._accelerator is not None:
            return self._accelerator.num_processes

        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()

        return self.config.world_size

    @property
    def rank(self) -> int:
        """Get current process rank."""
        if self.config.backend == DistributedBackend.NONE:
            return 0

        if self._accelerator is not None:
            return self._accelerator.process_index

        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()

        return self.config.rank

    @property
    def local_rank(self) -> int:
        """Get local rank (on current node)."""
        import os
        if self.config.backend == DistributedBackend.NONE:
            return 0

        # Try environment variable first (set by torchrun)
        env_local_rank = os.environ.get("LOCAL_RANK")
        if env_local_rank is not None:
            return int(env_local_rank)

        return self.config.local_rank

    @property
    def accelerator(self):
        """Get the underlying accelerate Accelerator (if using accelerate backend)."""
        return self._accelerator

    # === Model/Optimizer Preparation ===

    def prepare_model(
        self,
        model,
        optimizer: Optional[Any] = None,
        dataloader: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> tuple:
        """
        Prepare model and optimizer for distributed execution.

        Returns:
            tuple: (prepared_model, prepared_optimizer, prepared_dataloader, prepared_scheduler)
        """
        if self.config.backend == DistributedBackend.NONE:
            return model, optimizer, dataloader, lr_scheduler

        if self._accelerator is not None:
            return self._accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

        if self.config.backend == DistributedBackend.DEEPSPEED:
            return self._prepare_deepspeed(model, optimizer, dataloader)

        # For torch.distributed, wrap with DDP
        return self._prepare_ddp(model, optimizer, dataloader, lr_scheduler)

    def _prepare_ddp(
        self,
        model,
        optimizer: Optional[Any],
        dataloader: Optional[Any],
        lr_scheduler: Optional[Any],
    ) -> tuple:
        """Prepare model with torch.nn.parallel.DistributedDataParallel."""
        import torch
        import torch.nn.parallel as parallel

        # Move model to correct device
        if torch.cuda.is_available():
            model = model.to(f"cuda:{self.local_rank}")

        # Wrap with DDP
        model = parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
        )

        return model, optimizer, dataloader, lr_scheduler

    def _prepare_deepspeed(
        self,
        model,
        optimizer: Optional[Any],
        dataloader: Optional[Any],
    ) -> tuple:
        """Prepare model with DeepSpeed."""
        import deepspeed

        # DeepSpeed config - can be customized
        ds_config = {
            "train_batch_size": 16,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                }
            },
            "fp16": {
                "enabled": self.config.mixed_precision == "fp16"
            },
            "bf16": {
                "enabled": self.config.mixed_precision == "bf16"
            }
        }

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )

        return model_engine, optimizer, dataloader, None

    # === Synchronization ===

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.config.backend == DistributedBackend.NONE:
            return

        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            return

        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    def gather(self, tensor, destination: Optional[int] = None) -> List:
        """
        Gather tensors from all processes.

        Args:
            tensor: Tensor to gather
            destination: If specified, only that rank receives the result

        Returns:
            List of tensors from all processes (or None on non-destination ranks)
        """
        if self.config.backend == DistributedBackend.NONE:
            return [tensor]

        if self._accelerator is not None:
            return self._accelerator.gather(tensor)

        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)

        if destination is not None:
            if self.rank == destination:
                return gathered
            return None

        return gathered

    def reduce(self, tensor, op: str = "sum", destination: int = 0) -> Any:
        """
        Reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ("sum", "mean", "max", "min")
            destination: Rank that receives the result

        Returns:
            Reduced tensor (on destination rank) or None
        """
        if self.config.backend == DistributedBackend.NONE:
            return tensor

        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return tensor

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Divide by world_size after
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }

        dist.reduce(tensor, dst=destination, op=op_map.get(op, dist.ReduceOp.SUM))

        if self.rank == destination:
            if op == "mean":
                tensor /= self.world_size
            return tensor
        return None

    def broadcast(self, tensor, source: int = 0) -> Any:
        """
        Broadcast tensor from source to all processes.

        Args:
            tensor: Tensor to broadcast (only meaningful on source)
            source: Source rank

        Returns:
            Broadcasted tensor
        """
        if self.config.backend == DistributedBackend.NONE:
            return tensor

        import torch.distributed as dist

        if not dist.is_initialized():
            return tensor

        dist.broadcast(tensor, src=source)
        return tensor

    # === Context Managers ===

    @contextmanager
    def main_process_first(self):
        """Context manager to run code on main process first."""
        if self.config.backend == DistributedBackend.NONE:
            yield
            return

        if self.is_main_process:
            yield
        self.barrier()
        if not self.is_main_process:
            yield
        self.barrier()

    @contextmanager
    def local_main_process_first(self):
        """Context manager to run code on local main process first."""
        import os

        if self.config.backend == DistributedBackend.NONE:
            yield
            return

        is_local_main = self.local_rank == 0
        if is_local_main:
            yield
        self.barrier()
        if not is_local_main:
            yield
        self.barrier()

    # === Cleanup ===

    def destroy(self) -> None:
        """Clean up distributed resources."""
        if self.config.backend == DistributedBackend.TORCH_DISTRIBUTED:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("torch.distributed destroyed")
