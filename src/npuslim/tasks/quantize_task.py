"""Quantization task with streaming support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from npuslim.algorithms import BaseAlgorithm
from npuslim.core.backend import bh
from npuslim.registry import AlgorithmRegistry, TaskRegistry
from npuslim.streaming import SafeTensorStreamLoader, StreamSaver
from npuslim.tasks.base_task import BaseTask


@TaskRegistry.register("compressor", aliases=["QuantizeTask", "CompressorTask"])
class QuantizeTask(BaseTask):
    """
    Streaming quantization task.

    Loads model chunks, quantizes them, and saves incrementally.
    Enables quantizing 100B+ models on a single machine.
    """

    def __init__(
        self,
        *,
        name: str = "",
        model_path: Optional[str] = None,
        model_hub: str = "hf",
        block_name: str = "model.layers",
        device: str = "cpu",
        chunk_size: int = 1,
        calib_dataset: Optional[Dict[str, Any]] = None,
        algorithm: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model_path = Path(model_path) if model_path else None
        self.model_hub = model_hub
        self.block_name = block_name
        self.device = device
        self.chunk_size = max(int(chunk_size), 1)
        self.calib_config = calib_dataset
        self.algo_config = algorithm or {}
        self.saver_config = saver or {}

    def _create_loader(self) -> SafeTensorStreamLoader:
        """Create stream loader for model tensors."""
        if self.model_path is None:
            raise ValueError("model_path is required")

        return SafeTensorStreamLoader(
            model_path=self.model_path,
            model_hub=self.model_hub,
            block_name=self.block_name,
            tensor_device=self.device,
        )

    def _create_saver(self) -> Optional[StreamSaver]:
        """Create stream saver if output directory is configured."""
        output_dir = self.saver_config.get("output_dir") or self.saver_config.get("save_dir")
        if not output_dir:
            return None

        return StreamSaver(
            output_dir=Path(output_dir),
            shard_size=self.saver_config.get("shard_size", "5GB"),
            size_threshold=int(self.saver_config.get("size_threshold", 4 * 1024 * 1024 * 1024)),
        )

    def _create_algorithm(self) -> BaseAlgorithm:
        """Create quantization algorithm from config."""
        algo_type = self.algo_config.get("type")
        if not algo_type:
            raise ValueError("Algorithm type not specified")

        algo_cls = AlgorithmRegistry.get(algo_type)
        algo_kwargs = {k: v for k, v in self.algo_config.items() if k != "type"}
        return algo_cls(**algo_kwargs)

    def _load_calib_data(self) -> Optional[Any]:
        """Load calibration dataset if configured."""
        if not self.calib_config:
            return None
        # TODO: Implement dataset loading when needed
        logger.warning("Calibration dataset loading not yet implemented")
        return None

    def run(self) -> Dict[str, Any]:
        """Execute streaming quantization."""
        if self.model_path is None:
            raise ValueError("model_path is required")

        loader = self._create_loader()
        saver = self._create_saver()
        algo = self._create_algorithm()
        calib_data = self._load_calib_data()

        # Refresh loader index to get total layers
        loader.refresh_index()

        chunk_count = loader.get_chunk_count(self.chunk_size)
        logger.info(
            f"[QuantizeTask] Starting: chunks={chunk_count}, "
            f"chunk_size={self.chunk_size}, device={self.device}"
        )

        algo.on_start()
        try:
            for chunk_idx in range(chunk_count):
                # Load chunk
                chunk_layers = loader.load_chunk(chunk_idx, self.chunk_size)
                logger.info(f"[QuantizeTask] Chunk {chunk_idx}/{chunk_count}: {len(chunk_layers)} layers")

                # Process each layer in chunk
                for layer_info in chunk_layers:
                    tensors = layer_info.get("tensors", {})

                    # Quantize
                    quantized = algo.process_chunk(tensors, calib_data)

                    # Save if streaming
                    if saver is not None:
                        saver.add_tensors(quantized)
                    else:
                        # Update in-place for non-streaming mode
                        layer_info["tensors"] = quantized

                # Release chunk memory
                loader.unload_chunk(chunk_idx)
                bh.full_vacuum(self.device)

        finally:
            if saver is not None:
                saver.finalize()
                logger.success(f"[QuantizeTask] Saved to: {saver.output_dir}")

        algo.on_finish()
        logger.success("[QuantizeTask] Completed")

        return {
            "model_path": str(self.model_path),
            "chunks_processed": chunk_count,
            "output_dir": str(saver.output_dir) if saver else None,
        }
