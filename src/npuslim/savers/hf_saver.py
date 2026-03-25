# src/npuslim/savers/hf_saver.py
"""HuggingFace format saver."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger
from safetensors.torch import save_file

from npuslim.savers.base_saver import BaseSaver


class HuggingFaceSaver(BaseSaver):
    """Save tensors in HuggingFace safetensors format."""

    def __init__(
        self,
        output_dir: Path | str,
        size_threshold: int = 4 * 1024 * 1024 * 1024,  # 4 GiB
        shard_name_pattern: str = "model-{:05d}.safetensors",
    ):
        self.output_dir = Path(output_dir)
        self.size_threshold = int(size_threshold)
        self.shard_name_pattern = shard_name_pattern

        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_size: int = 0
        self.shard_counter: int = 0
        self.weight_map: Dict[str, str] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add tensor to buffer, auto-flush if threshold exceeded."""
        tensor_size = tensor.numel() * tensor.element_size()

        # Flush if adding would exceed threshold
        if self.buffer_size + tensor_size > self.size_threshold and self.buffer:
            self.flush()

        self.buffer[name] = tensor.cpu().contiguous()
        self.buffer_size += tensor_size

    def add_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Add multiple tensors."""
        for name, tensor in tensors.items():
            self.add_tensor(name, tensor)

    def flush(self) -> Optional[str]:
        """Write buffer to safetensors shard."""
        if not self.buffer:
            return None

        # Check disk space
        total, used, free = shutil.disk_usage(self.output_dir)
        if free < self.buffer_size * 1.1:
            raise IOError(f"Insufficient disk space: {free / 1e9:.2f} GiB free")

        shard_name = self.shard_name_pattern.format(self.shard_counter)
        shard_path = self.output_dir / shard_name

        save_file(self.buffer, shard_path)

        # Track weight map for index
        for name in self.buffer.keys():
            self.weight_map[name] = shard_name

        logger.info(f"[HFSaver] Flushed {len(self.buffer)} tensors to {shard_name}")

        # Clear buffer
        self.buffer.clear()
        self.buffer_size = 0
        self.shard_counter += 1

        return shard_name

    def finalize(self) -> None:
        """Flush remaining buffer and write index."""
        self.flush()

        # Build index
        total_size = sum(
            (self.output_dir / shard).stat().st_size
            for shard in set(self.weight_map.values())
            if (self.output_dir / shard).exists()
        )

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": self.weight_map,
        }

        index_path = self.output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.success(f"[HFSaver] Finalized: {len(self.weight_map)} tensors, index written")
