"""Streaming utilities for NPUSlim."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from npuslim.core.backend import bh


class SafeTensorIndex:
    """Reader for `model.safetensors.index.json`."""

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing safetensors index: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)

        weight_map = self._data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index format: {self.index_path}")

        self.weight_map: Dict[str, str] = weight_map
        self.total_size: int = int(self._data.get("metadata", {}).get("total_size", 0))

    def shard_for(self, tensor_name: str) -> str:
        """Get shard filename for a tensor."""
        if tensor_name not in self.weight_map:
            raise KeyError(f"Tensor '{tensor_name}' not found in index")
        return self.weight_map[tensor_name]

    def tensors_for_shard(self, shard_name: str) -> List[str]:
        """List tensor names in a specific shard."""
        return [name for name, shard in self.weight_map.items() if shard == shard_name]


class ShardTensorReader:
    """Lazy safetensors shard reader with open-handle cache."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.opened_shards: Dict[str, Any] = {}

    def _get_shard_handle(self, shard_name: str):
        if shard_name in self.opened_shards:
            return self.opened_shards[shard_name]

        shard_path = self.model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Safetensors shard not found: {shard_path}")

        handle = safe_open(shard_path, framework="pt", device=self.device)
        self.opened_shards[shard_name] = handle
        return handle

    def get_tensor(self, shard_name: str, tensor_name: str) -> torch.Tensor:
        """Read one tensor from shard by name."""
        handle = self._get_shard_handle(shard_name)
        return handle.get_tensor(tensor_name)

    def release_shard(self, shard_name: str) -> None:
        """Release one opened shard handle."""
        handle = self.opened_shards.pop(shard_name, None)
        if handle is None:
            return
        if hasattr(handle, "__exit__"):
            handle.__exit__(None, None, None)

    def clear(self) -> None:
        """Release all opened shard handles."""
        shard_names = list(self.opened_shards.keys())
        for shard_name in shard_names:
            self.release_shard(shard_name)


class StreamLoader:
    """Base loader interface for runtime chunk loading."""

    def __init__(self, tensor_device: str = "cpu"):
        self.tensor_device = tensor_device
        self._loaded_chunks: Dict[int, Any] = {}

    @abstractmethod
    def load_chunk(self, chunk_index: int, chunk_size: int) -> Any:
        """Load a chunk by index."""

    def unload_chunk(self, chunk_index: int) -> None:
        """Release a chunk from memory."""
        self._loaded_chunks.pop(chunk_index, None)
        bh.empty_cache(self.tensor_device)

    @abstractmethod
    def get_total_layers(self, total_layers_hint: Optional[int] = None) -> int:
        """Return total number of transformer layers."""

    def get_chunk_count(self, chunk_size: int, total_layers_hint: Optional[int] = None) -> int:
        """Compute total number of chunks for a given chunk size."""
        total_layers = self.get_total_layers(total_layers_hint=total_layers_hint)
        if total_layers <= 0:
            return 0
        size = max(int(chunk_size), 1)
        return (total_layers + size - 1) // size


class SafeTensorStreamLoader(StreamLoader):
    """Safetensors-based chunk loader with local/HF/ModelScope resolution."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_hub: str = "hf",
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        block_name: str = "model.layers",
        tensor_device: str = "cpu",
    ):
        super().__init__(tensor_device=tensor_device)
        self.model_path = Path(model_path)
        self.path_str = str(model_path)
        self.model_hub = model_hub
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.block_name = block_name
        self.tensor_device = tensor_device

        self._resolved_model_dir: Optional[Path] = (
            self.model_path if self.model_path.exists() else None
        )
        self._remote_snapshot_dir: Optional[Path] = None
        self._safetensor_index: Dict[str, Any] = {}
        self._weight_map: Dict[str, str] = {}
        self._layer_tensor_map: Dict[int, List[str]] = {}
        self._chunk_shards: Dict[int, set[str]] = {}
        self._tensor_reader: Optional[ShardTensorReader] = None
        self._total_layers: Optional[int] = None

    @property
    def safetensor_index(self) -> Dict[str, Any]:
        return self._safetensor_index

    @property
    def total_size(self) -> int:
        return int(self._safetensor_index.get("metadata", {}).get("total_size", 0))

    def set_block_name(self, block_name: str) -> None:
        self.block_name = block_name
        self._build_layer_tensor_map()

    def refresh_index(self, total_layers_hint: Optional[int] = None) -> None:
        index_path = self.resolve_file("model.safetensors.index.json")
        if index_path is None or not index_path.exists():
            self._safetensor_index = {}
            self._weight_map = {}
            self._layer_tensor_map = {}
            self._total_layers = int(total_layers_hint) if total_layers_hint is not None else None
            return

        index = SafeTensorIndex(index_path)
        self._resolved_model_dir = index_path.parent
        self._safetensor_index = {
            "metadata": {"total_size": index.total_size},
            "weight_map": index.weight_map,
        }
        self._weight_map = index.weight_map
        self._tensor_reader = ShardTensorReader(
            self._resolved_model_dir, device=self.tensor_device
        )
        self._build_layer_tensor_map()

        if self._total_layers is None and total_layers_hint is not None:
            self._total_layers = int(total_layers_hint)

    def resolve_file(self, filename: str) -> Optional[Path]:
        """Resolve file from local model directory or remote hub cache."""
        if self.model_path.exists():
            local_file = self.model_path / filename
            return local_file if local_file.exists() else None

        if self.model_hub == "hf":
            return self._resolve_remote_hf_file(filename)
        if self.model_hub == "ms":
            return self._resolve_remote_ms_file(filename)
        return None

    def resolve_model_source(self) -> str:
        """
        Resolve source path for `from_pretrained`.

        For ModelScope repo IDs, this returns a local snapshot path so
        downstream loaders can work with plain filesystem input.
        """
        if self.model_path.exists():
            return str(self.model_path)
        if self.model_hub == "ms":
            snapshot_dir = self._ensure_ms_snapshot_dir()
            if snapshot_dir is not None and snapshot_dir.exists():
                return str(snapshot_dir)
        return self.path_str

    def _resolve_remote_hf_file(self, filename: str) -> Optional[Path]:
        try:
            from huggingface_hub import hf_hub_download

            revision = self.tokenizer_kwargs.get("revision") or self.model_kwargs.get(
                "revision"
            )
            cached_file = hf_hub_download(
                repo_id=self.path_str,
                filename=filename,
                revision=revision,
            )
            return Path(cached_file)
        except Exception as exc:
            logger.debug(
                f"Failed to resolve remote file '{filename}' from HF Hub: {exc}"
            )
            return None

    def _ensure_ms_snapshot_dir(self) -> Optional[Path]:
        if self._remote_snapshot_dir is not None and self._remote_snapshot_dir.exists():
            return self._remote_snapshot_dir

        try:
            from modelscope.hub.snapshot_download import snapshot_download

            revision = self.tokenizer_kwargs.get("revision") or self.model_kwargs.get(
                "revision"
            )
            snapshot_dir = snapshot_download(
                model_id=self.path_str,
                revision=revision,
            )
            self._remote_snapshot_dir = Path(snapshot_dir)
            return self._remote_snapshot_dir
        except Exception as exc:
            logger.debug(
                f"Failed to download ModelScope snapshot for '{self.path_str}': {exc}"
            )
            return None

    def _resolve_remote_ms_file(self, filename: str) -> Optional[Path]:
        snapshot_dir = self._ensure_ms_snapshot_dir()
        if snapshot_dir is None:
            return None

        cached_file = snapshot_dir / filename
        if cached_file.exists():
            return cached_file
        logger.debug(f"ModelScope snapshot exists but file is missing: {cached_file}")
        return None

    def _build_layer_tensor_map(self) -> None:
        self._layer_tensor_map = {}
        max_idx = -1
        pattern = re.compile(rf"^{re.escape(self.block_name)}\.(\d+)\.")
        for tensor_name in self._weight_map:
            match = pattern.match(tensor_name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            self._layer_tensor_map.setdefault(layer_idx, []).append(tensor_name)
            max_idx = max(max_idx, layer_idx)

        if max_idx >= 0:
            self._total_layers = max_idx + 1

    def get_total_layers(self, total_layers_hint: Optional[int] = None) -> int:
        if self._total_layers is not None:
            return self._total_layers
        if total_layers_hint is not None:
            self._total_layers = int(total_layers_hint)
        return self._total_layers or 0

    def _ensure_reader(self) -> ShardTensorReader:
        if self._tensor_reader is None:
            self._tensor_reader = ShardTensorReader(
                self._resolved_model_dir or self.model_path,
                device=self.tensor_device,
            )
        return self._tensor_reader

    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        shard = self._weight_map.get(tensor_name)
        if shard is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in weight map")

        reader = self._ensure_reader()
        try:
            return reader.get_tensor(shard, tensor_name)
        except FileNotFoundError:
            shard_path = self.resolve_file(shard)
            if shard_path is None:
                raise
            if self._resolved_model_dir != shard_path.parent:
                self._resolved_model_dir = shard_path.parent
                self._tensor_reader = ShardTensorReader(
                    self._resolved_model_dir, device=self.tensor_device
                )
            return self._tensor_reader.get_tensor(shard, tensor_name)

    def load_chunk(self, chunk_index: int, chunk_size: int) -> List[Dict[str, Any]]:
        size = max(int(chunk_size), 1)
        start = chunk_index * size
        end = min(start + size, self.get_total_layers())
        payload: List[Dict[str, Any]] = []
        used_shards: set[str] = set()
        layer_count = max(end - start, 0)

        logger.info(
            f"[StreamLoader] Loading chunk {chunk_index}: layers {start}->{max(end - 1, start - 1)} "
            f"(count={layer_count}, device={self.tensor_device})"
        )

        layer_iter = tqdm(
            range(start, end),
            desc=f"chunk {chunk_index} load",
            disable=layer_count <= 1,
        )
        for layer_idx in layer_iter:
            layer_name = f"{self.block_name}.{layer_idx}"
            tensor_names = self._layer_tensor_map.get(layer_idx, [])
            for tensor_name in tensor_names:
                shard = self._weight_map.get(tensor_name)
                if shard:
                    used_shards.add(shard)
            tensors = {name: self._load_tensor(name) for name in tensor_names}
            payload.append({"name": layer_name, "index": layer_idx, "tensors": tensors})

        self._loaded_chunks[chunk_index] = payload
        self._chunk_shards[chunk_index] = used_shards
        logger.info(
            f"[StreamLoader] Loaded chunk {chunk_index}: layers={len(payload)}, shards={len(used_shards)}"
        )
        return payload

    def unload_chunk(self, chunk_index: int) -> None:
        super().unload_chunk(chunk_index)
        if self._tensor_reader is not None and chunk_index in self._chunk_shards:
            for shard_name in self._chunk_shards.pop(chunk_index):
                self._tensor_reader.release_shard(shard_name)

    def close(self) -> None:
        self._loaded_chunks.clear()
        self._chunk_shards.clear()
        if self._tensor_reader is not None:
            self._tensor_reader.clear()
            self._tensor_reader = None


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
