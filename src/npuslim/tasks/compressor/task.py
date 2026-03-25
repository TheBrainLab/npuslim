# src/npuslim/tasks/compressor/task.py
"""Compressor task for streaming quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from npuslim.algorithms import BaseAlgorithm
from npuslim.core.backend import bh
from npuslim.registry import AlgorithmRegistry, SaverRegistry, TaskRegistry
from npuslim.tasks.base_task import (
    BaseTask,
    RecipeTaskConfig,
    register_task_config,
)
from npuslim.tasks.compressor.context import ChunkContext
from npuslim.tasks.compressor.loader import ChunkLoader

if TYPE_CHECKING:
    from npuslim.core.resource_manager import ResourceManager
    from npuslim.savers.base_saver import BaseSaver


# =============================================================================
# Compressor-specific Configs (co-located with task)
# =============================================================================

@dataclass
class ExecutionConfig:
    """Compressor-specific execution options."""

    mode: str = "streaming"
    chunk_size: int = 1


@register_task_config("compressor", aliases=["CompressorTask", "QuantizeTask"])
@dataclass
class CompressorTaskConfig(RecipeTaskConfig):
    """Compressor/quantize task configuration with task-specific options."""

    ignore_layers: List[str] = field(default_factory=list)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


# =============================================================================
# Compressor Task Implementation
# =============================================================================

@TaskRegistry.register("compressor", aliases=["CompressorTask", "QuantizeTask"])
class CompressorTask(BaseTask):
    """
    Streaming compression task.

    - Receives resource_manager from engine
    - Acquires model/dataset via rm.acquire_*()
    - Loads chunks, applies algorithm, saves incrementally
    - Publishes updated model state for downstream tasks
    """

    def __init__(
        self,
        name: str = "",
        model: Optional[str] = None,
        data: Optional[str] = None,
        algorithm: Optional[Dict[str, Any]] = None,
        execution: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        resource_manager: Optional["ResourceManager"] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.model_ref = model  # e.g., "@qwen3"
        self.data_ref = data    # e.g., "@calib_data"
        self.rm = resource_manager

        self.algorithm_config = algorithm or {}
        self.execution_config = execution or {}
        self.saver_config = saver or {}

        # Execution settings
        self.mode = self.execution_config.get("mode", "streaming")
        self.chunk_size = max(int(self.execution_config.get("chunk_size", 1)), 1)
        self.device = self.execution_config.get("device", bh.default_device_str())

        # Extra settings from kwargs
        self.ignore_layers = kwargs.get("ignore_layers", [])
        self.block_name = kwargs.get("block_name", "model.layers")

    def _create_loader(self, model_obj) -> ChunkLoader:
        """Create chunk loader from model object."""
        return ChunkLoader(
            model_path=getattr(model_obj, "path_str", str(getattr(model_obj, "path", ""))),
            block_name=self.block_name,
            model_hub=getattr(model_obj, "model_hub", "hf"),
            tensor_device=self.device,
            chunk_size=self.chunk_size,
        )

    def _create_algorithm(self) -> BaseAlgorithm:
        """Create algorithm from config."""
        algo_type = self.algorithm_config.get("type")
        if not algo_type:
            raise ValueError("Algorithm type not specified")

        algo_kwargs = {k: v for k, v in self.algorithm_config.items() if k != "type"}
        algo_cls = AlgorithmRegistry.get(algo_type)
        return algo_cls(**algo_kwargs)

    def _create_saver(self) -> Optional["BaseSaver"]:
        """Create saver from config."""
        if not self.saver_config:
            return None

        saver_type = self.saver_config.get("type", "HuggingFaceSaver")
        saver_kwargs = {k: v for k, v in self.saver_config.items() if k != "type"}

        return SaverRegistry.create(saver_type, **saver_kwargs)

    def run(self) -> Dict[str, Any]:
        """
        Execute streaming compression.

        Acquires resources from resource_manager, processes chunks,
        saves results, and publishes updated model state.
        """
        if self.rm is None:
            raise ValueError("resource_manager is required")

        if self.model_ref is None:
            raise ValueError("model reference is required")

        # Acquire model from resource manager
        model = self.rm.acquire_model(self.model_ref)

        # Acquire calibration data if specified
        calib_data = None
        if self.data_ref:
            calib_data = self.rm.acquire_dataset(self.data_ref)

        # Create components
        loader = self._create_loader(model)
        algo = self._create_algorithm()
        saver = self._create_saver()

        loader.refresh_index()
        chunk_count = loader.get_chunk_count()

        logger.info(
            f"[CompressorTask] Starting: mode={self.mode}, "
            f"chunks={chunk_count}, chunk_size={self.chunk_size}, device={self.device}"
        )

        algo.on_start()
        try:
            for chunk_idx in range(chunk_count):
                # Load chunk
                chunk = loader.load_chunk(chunk_idx)
                chunk.calib_data = calib_data
                chunk.metadata["ignore_layers"] = self.ignore_layers

                # Apply algorithm
                chunk = algo.process_chunk(chunk)

                # Save if streaming
                if saver is not None:
                    saver.add_tensors(chunk.all_tensors())

                # Release memory
                loader.unload_chunk(chunk_idx)

        finally:
            if saver is not None:
                saver.finalize()
                logger.success(f"[CompressorTask] Saved to: {saver.output_dir}")

        algo.on_finish()
        loader.close()

        logger.success("[CompressorTask] Completed")

        # Publish updated model state for downstream tasks
        self.rm.publish_model_state(
            self.model_ref,
            model,
            state_meta={
                "quantized": True,
                "algorithm": self.algorithm_config.get("type"),
                "output_dir": str(saver.output_dir) if saver else None,
            },
        )

        return {
            "chunks_processed": chunk_count,
            "output_dir": str(saver.output_dir) if saver else None,
        }

    def execute(self) -> Dict[str, Any]:
        """Entry point called by engine."""
        self.on_start()
        try:
            return self.run()
        finally:
            self.on_finish()

