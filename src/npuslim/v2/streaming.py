# src/npuslim/v2/streaming.py
"""Streaming utilities for NPUSlim v2."""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from loguru import logger
from safetensors.torch import save_file

from npuslim.v2.config import ChunkConfig


class StreamLoader:
    """
    Loads model chunks on demand.
    Integrates with HuggingFace accelerate for lazy loading.
    """

    def __init__(self, model_path: Path, chunk_config: ChunkConfig):
        self.model_path = Path(model_path)
        self.chunk_config = chunk_config
        self._loaded_chunks: Dict[int, Any] = {}

    def load_chunk(self, chunk_index: int) -> Dict[str, torch.nn.Module]:
        """Load a specific chunk of layers into memory."""
        # Implementation depends on model architecture
        raise NotImplementedError("Subclass must implement load_chunk")

    def unload_chunk(self, chunk_index: int) -> None:
        """Release a chunk from memory."""
        if chunk_index in self._loaded_chunks:
            del self._loaded_chunks[chunk_index]
            torch.cuda.empty_cache()

    def get_chunk_layers(self, chunk_index: int) -> List[str]:
        """Get layer names for a chunk."""
        # Implementation depends on model architecture
        raise NotImplementedError("Subclass must implement get_chunk_layers")


class StreamSaver:
    """
    Saves quantized tensors to safetensors shards.
    Buffers tensors and flushes when size threshold exceeded.
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size: str = "5GB",
        size_threshold: int = 4 * 1024 * 1024 * 1024,  # 4 GiB
    ):
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size  # TODO: Use for user-friendly size parsing (e.g., "5GB")
        self.size_threshold = size_threshold

        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_size: int = 0
        self.shard_counter: int = 0
        self.metadata: Dict[str, Dict] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add a tensor to buffer, flush if threshold exceeded."""
        tensor_size = tensor.numel() * tensor.element_size()

        # Check if we need to flush before adding
        if self.buffer_size + tensor_size > self.size_threshold and self.buffer:
            self.flush()

        self.buffer[name] = tensor.cpu().contiguous()
        self.buffer_size += tensor_size

    def add_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Add multiple tensors at once."""
        for name, tensor in tensors.items():
            self.add_tensor(name, tensor)

    def flush(self) -> Optional[str]:
        """Write buffer to a safetensors shard."""
        if not self.buffer:
            return None

        # Check disk space
        total, used, free = shutil.disk_usage(self.output_dir)
        if free < self.buffer_size * 1.1:
            raise IOError(f"Insufficient disk space: {free / 1e9:.2f} GiB free")

        shard_name = f"model-{self.shard_counter:05d}.safetensors"
        shard_path = self.output_dir / shard_name

        save_file(self.buffer, shard_path)

        # Track metadata
        self.metadata[shard_name] = {
            "weight_map": {name: shard_name for name in self.buffer.keys()},
        }

        logger.info(f"Flushed {len(self.buffer)} tensors ({self.buffer_size / 1e6:.2f} MB) to {shard_name}")

        # Clear buffer
        self.buffer.clear()
        self.buffer_size = 0
        self.shard_counter += 1

        return shard_name

    def finalize(self, model_config: Any = None, tokenizer: Any = None) -> None:
        """Finalize: flush remaining, save index, save config/tokenizer."""
        # Flush remaining buffer
        self.flush()

        # Build and save index
        index = self._build_index()
        index_path = self.output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        # Save config
        if model_config and hasattr(model_config, "save_pretrained"):
            model_config.save_pretrained(self.output_dir)

        # Save tokenizer
        if tokenizer and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(self.output_dir)

        logger.success(f"Streaming save finalized: {self.output_dir}")

    def _build_index(self) -> Dict:
        """Build model.safetensors.index.json."""
        weight_map = {}
        for shard_name, meta in self.metadata.items():
            weight_map.update(meta["weight_map"])

        total_size = sum(
            (self.output_dir / shard_name).stat().st_size
            for shard_name in self.metadata
            if (self.output_dir / shard_name).exists()
        )

        return {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
