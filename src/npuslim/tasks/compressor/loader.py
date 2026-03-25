# src/npuslim/tasks/compressor/loader.py
"""Chunk loader for streaming tensor loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from loguru import logger
from safetensors import safe_open

from npuslim.core.backend import bh
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


class ChunkLoader:
    """Streaming loader for transformer layer chunks."""

    def __init__(
        self,
        model_path: str | Path,
        block_name: str = "model.layers",
        model_hub: str = "hf",
        tensor_device: str = "cpu",
        chunk_size: int = 1,
    ):
        self.model_path = Path(model_path)
        self.block_name = block_name
        self.model_hub = model_hub
        self.tensor_device = tensor_device
        self.chunk_size = max(int(chunk_size), 1)

        self._resolved_dir: Optional[Path] = None
        self._weight_map: Dict[str, str] = {}
        self._layer_tensor_map: Dict[int, List[str]] = {}
        self._total_layers: Optional[int] = None
        self._opened_shards: Dict[str, Any] = {}

    def refresh_index(self, total_layers_hint: Optional[int] = None) -> None:
        """Refresh safetensors index."""
        index_path = self._resolve_file("model.safetensors.index.json")

        if index_path and index_path.exists():
            self._load_index_file(index_path)
        else:
            # Fallback to single safetensors file
            shard_path = self._resolve_file("model.safetensors")
            if shard_path and shard_path.exists():
                self._load_single_shard(shard_path)
            else:
                logger.warning(f"No safetensors found in {self.model_path}")
                self._total_layers = total_layers_hint

        if total_layers_hint is not None:
            self._total_layers = int(total_layers_hint)

    def _load_index_file(self, index_path: Path) -> None:
        """Load model.safetensors.index.json."""
        with open(index_path) as f:
            data = json.load(f)

        self._resolved_dir = index_path.parent
        self._weight_map = data.get("weight_map", {})
        self._build_layer_tensor_map()
        logger.info(f"[ChunkLoader] Loaded index: {len(self._weight_map)} tensors")

    def _load_single_shard(self, shard_path: Path) -> None:
        """Fallback: load from single model.safetensors."""
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                tensor_names = list(handle.keys())

            shard_name = shard_path.name
            self._resolved_dir = shard_path.parent
            self._weight_map = {name: shard_name for name in tensor_names}
            self._build_layer_tensor_map()
            logger.info(f"[ChunkLoader] Single shard: {len(tensor_names)} tensors")
        except Exception as e:
            logger.warning(f"[ChunkLoader] Failed to load single shard: {e}")

    def _build_layer_tensor_map(self) -> None:
        """Build mapping from layer index to tensor names."""
        self._layer_tensor_map = {}
        max_idx = -1
        pattern = re.compile(rf"^{re.escape(self.block_name)}\.(\d+)\.")

        for tensor_name in self._weight_map:
            match = pattern.match(tensor_name)
            if match:
                layer_idx = int(match.group(1))
                self._layer_tensor_map.setdefault(layer_idx, []).append(tensor_name)
                max_idx = max(max_idx, layer_idx)

        if max_idx >= 0:
            self._total_layers = max_idx + 1

    def _resolve_file(self, filename: str) -> Optional[Path]:
        """Resolve file from local path or hub."""
        if self.model_path.exists():
            local_file = self.model_path / filename
            return local_file if local_file.exists() else None

        # TODO: Add HF Hub and ModelScope resolution
        return None

    def _get_shard_handle(self, shard_name: str):
        """Get or open shard handle."""
        if shard_name in self._opened_shards:
            return self._opened_shards[shard_name]

        if self._resolved_dir is None:
            raise RuntimeError("Index not loaded. Call refresh_index() first.")

        shard_path = self._resolved_dir / shard_name
        handle = safe_open(shard_path, framework="pt", device=self.tensor_device)
        self._opened_shards[shard_name] = handle
        return handle

    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a single tensor by name."""
        shard = self._weight_map.get(tensor_name)
        if shard is None:
            raise KeyError(f"Tensor not found: {tensor_name}")

        handle = self._get_shard_handle(shard)
        return handle.get_tensor(tensor_name)

    def get_total_layers(self) -> int:
        """Get total number of transformer layers."""
        return self._total_layers or 0

    def get_chunk_count(self) -> int:
        """Get number of chunks."""
        total = self.get_total_layers()
        if total <= 0:
            return 0
        return (total + self.chunk_size - 1) // self.chunk_size

    def load_chunk(self, chunk_index: int) -> ChunkContext:
        """Load a chunk of layers."""
        start = chunk_index * self.chunk_size
        end = min(start + self.chunk_size, self.get_total_layers())

        layers: List[LayerInfo] = []
        for layer_idx in range(start, end):
            layer_name = f"{self.block_name}.{layer_idx}"
            tensor_names = self._layer_tensor_map.get(layer_idx, [])

            tensors: Dict[str, torch.Tensor] = {}
            for full_name in tensor_names:
                # Strip layer prefix to get relative tensor name
                rel_name = full_name[len(layer_name) + 1:]
                tensors[rel_name] = self._load_tensor(full_name)

            layers.append(LayerInfo(name=layer_name, index=layer_idx, tensors=tensors))

        logger.info(f"[ChunkLoader] Loaded chunk {chunk_index}: {len(layers)} layers")
        return ChunkContext(chunk_index=chunk_index, layers=layers)

    def unload_chunk(self, chunk_index: int) -> None:
        """Release chunk memory."""
        # Release shard handles that were opened for this chunk
        # For simplicity, release all (can be optimized later)
        for shard_name in list(self._opened_shards.keys()):
            handle = self._opened_shards.pop(shard_name)
            if hasattr(handle, "__exit__"):
                handle.__exit__(None, None, None)

        bh.full_vacuum(self.tensor_device)
        logger.debug(f"[ChunkLoader] Unloaded chunk {chunk_index}")

    def __iter__(self) -> Iterator[ChunkContext]:
        """Iterate over all chunks."""
        for i in range(self.get_chunk_count()):
            yield self.load_chunk(i)
            self.unload_chunk(i)

    def close(self) -> None:
        """Clean up all resources."""
        self._opened_shards.clear()
        self._layer_tensor_map.clear()
