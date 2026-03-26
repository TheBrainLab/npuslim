# src/npuslim/tasks/compressor/loader.py
"""Chunk loader for streaming tensor loading with hub support."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm

from npuslim.core.backend import bh
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo, ModuleInfo


class ChunkLoader:
    """
    Streaming loader that divides transformer layers into chunks and
    optionally loads pre/post-transformer modules.

    Supports:
    - Local model directories
    - HuggingFace Hub (model_id like "Qwen/Qwen3-0.6B")
    - ModelScope Hub (model_id with model_hub="ms")
    """

    def __init__(
        self,
        model_path: str | Path,
        model_hub: str = "hf",
        model_kwargs: Optional[Dict[str, Any]] = None,
        tensor_device: str = "cpu",
        chunk_size: int = 1,
        block_name: str = "model.layers",
        pre_module_names: Optional[List[str]] = None,
        post_module_names: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        self.path_str = str(model_path)
        self.model_hub = model_hub
        self.model_kwargs = model_kwargs or {}
        self.tensor_device = tensor_device
        self.chunk_size = max(int(chunk_size), 1)
        self.block_name = block_name
        self.pre_module_names = [name for name in (pre_module_names or []) if name]
        self.post_module_names = [name for name in (post_module_names or []) if name]

        # Resolved state
        self._resolved_dir: Optional[Path] = None
        self._remote_snapshot_dir: Optional[Path] = None

        # Index data
        self._weight_map: Dict[str, str] = {}  # tensor_name -> shard_name
        self._tensor_names: List[str] = []  # ordered list of all tensor names
        self._layer_tensor_map: Dict[int, List[str]] = {}  # layer_idx -> tensor_names
        self._layer_indices: List[int] = []  # sorted layer indices in inference order
        self._pre_module_tensor_map: Dict[str, List[str]] = {}
        self._post_module_tensor_map: Dict[str, List[str]] = {}
        self._unassigned_tensor_names: List[str] = []
        self._checkpoint_format: str = "unknown"  # safetensors | torch_bin

        # Opened shard cache
        self._opened_shards: Dict[str, Any] = {}

    def refresh_index(self) -> None:
        """Refresh checkpoint index from local or remote source."""
        # 1) Safetensors sharded index
        st_index = self._resolve_file("model.safetensors.index.json")
        if st_index and st_index.exists():
            self._load_index_file(st_index, checkpoint_format="safetensors")
            return

        # 2) Safetensors single shard
        st_single = self._resolve_file("model.safetensors")
        if st_single and st_single.exists():
            self._load_single_safetensors_shard(st_single)
            return

        # 3) PyTorch bin sharded index
        pt_index = self._resolve_file("pytorch_model.bin.index.json")
        if pt_index and pt_index.exists():
            self._load_index_file(pt_index, checkpoint_format="torch_bin")
            return

        # 4) PyTorch single shard
        pt_single = self._resolve_file("pytorch_model.bin")
        if pt_single and pt_single.exists():
            self._load_single_torch_bin_shard(pt_single)
            return

        self._weight_map = {}
        self._tensor_names = []
        self._layer_tensor_map = {}
        self._layer_indices = []
        self._pre_module_tensor_map = {}
        self._post_module_tensor_map = {}
        self._unassigned_tensor_names = []
        self._checkpoint_format = "unknown"
        logger.warning(
            f"[ChunkLoader] No supported checkpoint found for {self.path_str}. "
            "Tried: model.safetensors(.index.json), pytorch_model.bin(.index.json)"
        )

    def _load_index_file(self, index_path: Path, checkpoint_format: str) -> None:
        """Load sharded checkpoint index json (safetensors or pytorch bin)."""
        with open(index_path) as f:
            data = json.load(f)

        self._resolved_dir = index_path.parent
        self._weight_map = data.get("weight_map", {})
        self._checkpoint_format = checkpoint_format
        self._tensor_names = list(self._weight_map.keys())
        self._build_layer_tensor_map()
        self._build_aux_tensor_lists()
        self._validate_tensor_assignment()
        logger.info(
            f"[ChunkLoader] Loaded {checkpoint_format} index: {len(self._tensor_names)} tensors"
        )

    def _load_single_safetensors_shard(self, shard_path: Path) -> None:
        """Fallback: load from single model.safetensors."""
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                tensor_names = list(handle.keys())

            shard_name = shard_path.name
            self._resolved_dir = shard_path.parent
            self._weight_map = {name: shard_name for name in tensor_names}
            self._checkpoint_format = "safetensors"
            self._tensor_names = list(tensor_names)
            self._build_layer_tensor_map()
            self._build_aux_tensor_lists()
            self._validate_tensor_assignment()
            logger.info(f"[ChunkLoader] Single shard: {len(tensor_names)} tensors")
        except Exception as e:
            logger.warning(f"[ChunkLoader] Failed to load single safetensors shard: {e}")

    @staticmethod
    def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict checkpoint, got {type(obj).__name__}")

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            candidate = obj["state_dict"]
        else:
            candidate = obj

        tensor_items = {k: v for k, v in candidate.items() if torch.is_tensor(v)}
        if not tensor_items:
            raise ValueError("No tensor entries found in checkpoint payload")
        return tensor_items

    def _torch_load_file(self, shard_path: Path) -> Dict[str, torch.Tensor]:
        try:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            # torch<2.0 may not support weights_only
            loaded = torch.load(shard_path, map_location="cpu")
        return self._extract_state_dict(loaded)

    def _load_single_torch_bin_shard(self, shard_path: Path) -> None:
        """Fallback: load from single pytorch_model.bin."""
        try:
            tensor_map = self._torch_load_file(shard_path)

            shard_name = shard_path.name
            self._resolved_dir = shard_path.parent
            self._weight_map = {name: shard_name for name in tensor_map.keys()}
            self._checkpoint_format = "torch_bin"
            self._tensor_names = list(tensor_map.keys())
            self._build_layer_tensor_map()
            self._build_aux_tensor_lists()
            self._validate_tensor_assignment()
            logger.info(f"[ChunkLoader] Single torch bin shard: {len(tensor_map)} tensors")
        except Exception as e:
            logger.warning(f"[ChunkLoader] Failed to load single torch bin shard: {e}")

    def _build_layer_tensor_map(self) -> None:
        """Build mapping from layer index to layer tensor names."""
        pattern = re.compile(rf"^{re.escape(self.block_name)}\.(\d+)\.")
        layer_tensor_map: Dict[int, List[str]] = {}

        for tensor_name in self._tensor_names:
            match = pattern.match(tensor_name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            layer_tensor_map.setdefault(layer_idx, []).append(tensor_name)

        self._layer_tensor_map = layer_tensor_map
        self._layer_indices = sorted(layer_tensor_map.keys())

    @staticmethod
    def _tensor_in_module(tensor_name: str, module_name: str) -> bool:
        if tensor_name == module_name:
            return True
        return tensor_name.startswith(f"{module_name}.")

    def _collect_module_tensor_map(self, module_names: List[str]) -> Dict[str, List[str]]:
        module_tensor_map: Dict[str, List[str]] = {name: [] for name in module_names}
        consumed: set[str] = set()

        for module_name in module_names:
            matched: List[str] = []
            for tensor_name in self._tensor_names:
                if tensor_name in consumed:
                    continue
                if self._tensor_in_module(tensor_name, module_name):
                    matched.append(tensor_name)
            if matched:
                module_tensor_map[module_name] = matched
                consumed.update(matched)
            else:
                module_tensor_map.pop(module_name, None)
        return module_tensor_map

    def _build_aux_tensor_lists(self) -> None:
        self._pre_module_tensor_map = self._collect_module_tensor_map(self.pre_module_names)
        self._post_module_tensor_map = self._collect_module_tensor_map(self.post_module_names)

    def _compute_unassigned_tensor_names(self) -> List[str]:
        assigned: set[str] = set()
        for tensor_names in self._layer_tensor_map.values():
            assigned.update(tensor_names)
        for tensor_names in self._pre_module_tensor_map.values():
            assigned.update(tensor_names)
        for tensor_names in self._post_module_tensor_map.values():
            assigned.update(tensor_names)
        return [tensor_name for tensor_name in self._tensor_names if tensor_name not in assigned]

    def _validate_tensor_assignment(self) -> None:
        """Ensure all tensors are assigned to pre/layers/post buckets."""
        self._unassigned_tensor_names = self._compute_unassigned_tensor_names()

        if self._unassigned_tensor_names:
            preview = ", ".join(self._unassigned_tensor_names[:8])
            if len(self._unassigned_tensor_names) > 8:
                preview += ", ..."
            logger.warning(
                "[ChunkLoader] Found "
                f"{len(self._unassigned_tensor_names)} unassigned tensors. "
                f"Examples: {preview}. "
                "These tensors will be preserved by CompressorTask backfill."
            )

    def _load_module_infos(self, module_tensor_map: Dict[str, List[str]]) -> List[ModuleInfo]:
        modules: List[ModuleInfo] = []
        for module_name, full_tensor_names in module_tensor_map.items():
            module_tensors: Dict[str, torch.Tensor] = {}
            module_prefix = f"{module_name}."
            for full_tensor_name in full_tensor_names:
                tensor = self._load_tensor(full_tensor_name)
                rel_tensor_name = (
                    full_tensor_name[len(module_prefix):]
                    if full_tensor_name.startswith(module_prefix)
                    else full_tensor_name
                )
                module_tensors[rel_tensor_name] = tensor
            modules.append(ModuleInfo(name=module_name, tensors=module_tensors))
        return modules

    def _resolve_file(self, filename: str) -> Optional[Path]:
        """Resolve file from local path or hub."""
        if self.model_path.exists():
            local_file = self.model_path / filename
            return local_file if local_file.exists() else None

        if self.model_hub == "hf":
            return self._resolve_hf_file(filename)
        if self.model_hub == "ms":
            return self._resolve_ms_file(filename)

        return None

    def _resolve_hf_file(self, filename: str) -> Optional[Path]:
        """Resolve file from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            revision = self.model_kwargs.get("revision")
            cached_file = hf_hub_download(
                repo_id=self.path_str,
                filename=filename,
                revision=revision,
            )
            return Path(cached_file)
        except Exception as exc:
            logger.debug(f"[ChunkLoader] HF Hub resolve failed for '{filename}': {exc}")
            return None

    def _ensure_ms_snapshot_dir(self) -> Optional[Path]:
        """Ensure ModelScope snapshot is downloaded."""
        if self._remote_snapshot_dir is not None and self._remote_snapshot_dir.exists():
            return self._remote_snapshot_dir

        try:
            from modelscope.hub.snapshot_download import snapshot_download

            revision = self.model_kwargs.get("revision")
            snapshot_dir = snapshot_download(
                model_id=self.path_str,
                revision=revision,
            )
            self._remote_snapshot_dir = Path(snapshot_dir)
            return self._remote_snapshot_dir
        except Exception as exc:
            logger.debug(f"[ChunkLoader] ModelScope snapshot failed for '{self.path_str}': {exc}")
            return None

    def _resolve_ms_file(self, filename: str) -> Optional[Path]:
        """Resolve file from ModelScope Hub."""
        snapshot_dir = self._ensure_ms_snapshot_dir()
        if snapshot_dir is None:
            return None

        cached_file = snapshot_dir / filename
        if cached_file.exists():
            return cached_file
        logger.debug(f"[ChunkLoader] ModelScope snapshot exists but file missing: {cached_file}")
        return None

    def resolve_model_source(self) -> str:
        """Resolve source path for from_pretrained."""
        if self.model_path.exists():
            return str(self.model_path)
        if self.model_hub == "ms":
            snapshot_dir = self._ensure_ms_snapshot_dir()
            if snapshot_dir is not None and snapshot_dir.exists():
                return str(snapshot_dir)
        return self.path_str

    def _get_shard_handle(self, shard_name: str):
        """Get or open shard handle with caching."""
        if shard_name in self._opened_shards:
            return self._opened_shards[shard_name]

        if self._resolved_dir is None:
            raise RuntimeError("Index not loaded. Call refresh_index() first.")

        shard_path = self._resolved_dir / shard_name
        if self._checkpoint_format == "safetensors":
            handle = safe_open(shard_path, framework="pt", device=self.tensor_device)
        elif self._checkpoint_format == "torch_bin":
            handle = self._torch_load_file(shard_path)
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {self._checkpoint_format}")
        self._opened_shards[shard_name] = handle
        return handle

    def _load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a single tensor by name."""
        shard = self._weight_map.get(tensor_name)
        if shard is None:
            raise KeyError(f"Tensor not found: {tensor_name}")

        handle = self._get_shard_handle(shard)
        if self._checkpoint_format == "safetensors":
            return handle.get_tensor(tensor_name)

        if self._checkpoint_format == "torch_bin":
            tensor = handle.get(tensor_name)
            if not torch.is_tensor(tensor):
                raise KeyError(f"Tensor '{tensor_name}' not found in shard '{shard}'")
            if self.tensor_device != "cpu":
                return tensor.to(self.tensor_device)
            return tensor

        raise RuntimeError(f"Unsupported checkpoint format: {self._checkpoint_format}")

    # === Public API ===

    def get_total_tensors(self) -> int:
        """Get total number of tensors."""
        return len(self._tensor_names)

    def get_all_tensor_names(self) -> List[str]:
        """Get all original tensor names from safetensors index."""
        return list(self._tensor_names)

    def load_tensors(self, tensor_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load tensors by original tensor names.

        Args:
            tensor_names: Original tensor names from safetensors index.

        Returns:
            Mapping of tensor_name -> tensor.
        """
        tensors: Dict[str, torch.Tensor] = {}
        for tensor_name in tensor_names:
            tensors[tensor_name] = self._load_tensor(tensor_name)
        return tensors

    def get_total_layers(self) -> int:
        """Get total number of transformer layers."""
        return len(self._layer_indices)

    def get_chunk_count(self) -> int:
        """Get number of layer chunks."""
        total_layers = self.get_total_layers()
        if total_layers <= 0:
            return 0
        return (total_layers + self.chunk_size - 1) // self.chunk_size

    def _load_layers(self, layer_indices: List[int], progress_desc: str) -> List[LayerInfo]:
        layers: List[LayerInfo] = []
        layer_iter = tqdm(
            layer_indices,
            total=len(layer_indices),
            desc=progress_desc,
            disable=len(layer_indices) <= 1,
        )
        for layer_idx in layer_iter:
            layer_name = f"{self.block_name}.{layer_idx}"
            layer_tensors: Dict[str, torch.Tensor] = {}
            for full_tensor_name in self._layer_tensor_map.get(layer_idx, []):
                tensor = self._load_tensor(full_tensor_name)
                layer_prefix = f"{layer_name}."
                rel_tensor_name = (
                    full_tensor_name[len(layer_prefix):]
                    if full_tensor_name.startswith(layer_prefix)
                    else full_tensor_name
                )
                layer_tensors[rel_tensor_name] = tensor

            layers.append(
                LayerInfo(
                    name=layer_name,
                    index=layer_idx,
                    tensors=layer_tensors,
                )
            )
        return layers

    def load_full(self) -> ChunkContext:
        """Load full model tensors (pre + all transformer layers + post) in one pass."""
        layer_indices = list(self._layer_indices)
        pre_modules = self._load_module_infos(self._pre_module_tensor_map)
        layers = self._load_layers(layer_indices, progress_desc="full load")
        post_modules = self._load_module_infos(self._post_module_tensor_map)

        logger.info(
            f"[ChunkLoader] Loaded full model: pre_modules={len(pre_modules)}, "
            f"layers={len(layers)}, post_modules={len(post_modules)}, "
            f"tensors={sum(len(module.tensors) for module in pre_modules) + sum(len(layer.tensors) for layer in layers) + sum(len(module.tensors) for module in post_modules)}"
        )
        return ChunkContext(
            chunk_index=0,
            layers=layers,
            pre_modules=pre_modules,
            post_modules=post_modules,
        )

    def load_chunk(self, chunk_index: int) -> ChunkContext:
        """Load a chunk containing chunk_size layers."""
        start = chunk_index * self.chunk_size
        end = min(start + self.chunk_size, self.get_total_layers())
        layer_indices = self._layer_indices[start:end]
        is_first_chunk = chunk_index == 0
        is_last_chunk = chunk_index == max(self.get_chunk_count() - 1, 0)

        pre_modules: List[ModuleInfo] = []
        post_modules: List[ModuleInfo] = []

        if is_first_chunk and self._pre_module_tensor_map:
            pre_modules = self._load_module_infos(self._pre_module_tensor_map)
        layers = self._load_layers(layer_indices, progress_desc=f"chunk {chunk_index} load")

        if is_last_chunk and self._post_module_tensor_map:
            post_modules = self._load_module_infos(self._post_module_tensor_map)

        logger.info(
            f"[ChunkLoader] Loaded chunk {chunk_index}: "
            f"pre_modules={len(pre_modules)}, layers={len(layers)}, post_modules={len(post_modules)}, "
            f"tensors={sum(len(module.tensors) for module in pre_modules) + sum(len(layer.tensors) for layer in layers) + sum(len(module.tensors) for module in post_modules)}, "
            f"layer_range={layer_indices[0] if layer_indices else None}:{layer_indices[-1] if layer_indices else None}"
        )

        return ChunkContext(
            chunk_index=chunk_index,
            layers=layers,
            pre_modules=pre_modules,
            post_modules=post_modules,
        )

    def unload_chunk(self, chunk_index: int) -> None:
        """Release chunk memory."""
        for shard_name in list(self._opened_shards.keys()):
            handle = self._opened_shards.pop(shard_name)
            if hasattr(handle, "__exit__"):
                handle.__exit__(None, None, None)

        bh.empty_cache(self.tensor_device)
        logger.debug(f"[ChunkLoader] Unloaded chunk {chunk_index}")

    def __iter__(self) -> Iterator[ChunkContext]:
        """Iterate over all chunks."""
        for i in range(self.get_chunk_count()):
            yield self.load_chunk(i)
            self.unload_chunk(i)

    def close(self) -> None:
        """Clean up all resources."""
        self._opened_shards.clear()
        self._tensor_names.clear()
        self._weight_map.clear()
        self._layer_tensor_map.clear()
        self._layer_indices.clear()
        self._pre_module_tensor_map.clear()
        self._post_module_tensor_map.clear()
        self._unassigned_tensor_names.clear()
