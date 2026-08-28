"""HuggingFace-format streaming saver."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Optional

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from npuslim.core.backend import bh
from npuslim.core import SaverRegistry
from npuslim.savers.base_saver import BaseSaver


@SaverRegistry.register(
    "StreamingHuggingFaceSaver",
    aliases=["streaming_hf", "StreamingHFSaver", "streaming_hf_saver"],
)
class StreamingHuggingFaceSaver(BaseSaver):
    """Streaming safetensors saver with HuggingFace-compatible output layout."""

    _WEIGHT_FILE_SUFFIXES = (
        ".safetensors",
        ".bin",
        ".h5",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".msgpack",
    )
    _SKIP_SOURCE_FILE_NAMES = {
        "model.safetensors.index.json",
        "quant_model_description.json",
        "optimizer.pt",
        "training_args.bin",
        "trainer_state.json",
    }
    _SIZE_UNITS = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }

    def __init__(
        self,
        save_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        save_dir: Path | str | None = None,
        size_threshold: int | str = 4 * 1024 * 1024 * 1024,  # 4 GiB
        max_shard_size: int | str | None = None,
        shard_size: int | str | None = None,
        shard_name_pattern: str = "model-{:05d}.safetensors",
        copy_aux_files: bool = True,
        strip_quantization_config_on_npu: bool = True,
        require_tensor_types_on_npu: bool = True,
    ):
        super().__init__(
            save_path=save_path,
            output_dir=output_dir,
            save_dir=save_dir,
        )

        threshold_source = max_shard_size if max_shard_size is not None else shard_size
        if threshold_source is None:
            threshold_source = size_threshold

        self.size_threshold = self._parse_size_to_bytes(threshold_source)
        self.shard_name_pattern = shard_name_pattern
        self.copy_aux_files = bool(copy_aux_files)
        self.strip_quantization_config_on_npu = bool(strip_quantization_config_on_npu)
        self.require_tensor_types_on_npu = bool(require_tensor_types_on_npu)

        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_size: int = 0
        self.shard_counter: int = 0
        self.weight_map: Dict[str, str] = {}
        self.tensor_type_map: Dict[str, str] = {}
        self._written_shards: set[str] = set()

        self._source_ref: Optional[str] = None
        self._source_model_hub: str = "hf"
        self._source_model_kwargs: Dict[str, Any] = {}
        self._source_dir: Optional[Path] = None
        self._model_config: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _parse_size_to_bytes(cls, value: int | str) -> int:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"size_threshold must be > 0, got {value}")
            return value

        value = str(value).strip().lower()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?b)?", value)
        if not match:
            raise ValueError(
                f"Invalid size value '{value}'. Examples: 1073741824, '1GB', '512MB'"
            )

        number = float(match.group(1))
        unit = match.group(2) or "b"
        multiplier = cls._SIZE_UNITS.get(unit)
        if multiplier is None:
            raise ValueError(f"Unsupported size unit '{unit}' in '{value}'")

        size = int(number * multiplier)
        if size <= 0:
            raise ValueError(f"size_threshold must be > 0, got {value}")
        return size

    def set_source(
        self,
        source: str | Path | None,
        *,
        model_hub: str = "hf",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set source model path/repo for auxiliary-file synchronization."""
        if not source:
            return

        self._source_model_hub = model_hub
        self._source_model_kwargs = dict(model_kwargs or {})
        self._source_ref = str(source)

        source_path = Path(str(source))
        if source_path.exists():
            self._source_dir = source_path

    def set_hf_assets(
        self,
        *,
        model_config: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
    ) -> None:
        """Attach already-loaded HF assets so saver can call `save_pretrained`."""
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._processor = processor

    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        tensor_type: Optional[str] = None,
    ) -> None:
        """Add tensor to buffer, auto-flush if threshold exceeded."""
        if bh.has_npu and self.require_tensor_types_on_npu and not tensor_type:
            raise ValueError(
                f"[HFSaver] NPU mode requires explicit tensor_type for '{name}'."
            )

        if name in self.buffer:
            old_tensor = self.buffer[name]
            self.buffer_size -= old_tensor.numel() * old_tensor.element_size()

        tensor_size = tensor.numel() * tensor.element_size()

        # Flush if adding would exceed threshold
        if self.buffer_size + tensor_size > self.size_threshold and self.buffer:
            self.flush()

        self.buffer[name] = tensor.cpu().contiguous()
        self.buffer_size += tensor_size
        if tensor_type:
            self.tensor_type_map[name] = str(tensor_type)
        elif name not in self.tensor_type_map:
            self.tensor_type_map[name] = "FLOAT"

        # Flush immediately once threshold is reached.
        if self.buffer_size >= self.size_threshold:
            self.flush()

    def add_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        tensor_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add multiple tensors."""
        tensor_types = tensor_types or {}
        for name, tensor in tensors.items():
            self.add_tensor(name, tensor, tensor_type=tensor_types.get(name))

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
        # Atomic write: crash mid-flush must not leave a corrupt shard file.
        tmp_path = shard_path.with_name(shard_name + ".tmp")
        save_file(self.buffer, str(tmp_path))
        os.replace(tmp_path, shard_path)
        self._written_shards.add(shard_name)

        # Track weight map for index
        for name in self.buffer.keys():
            self.weight_map[name] = shard_name

        logger.info(f"[HFSaver] Flushed {len(self.buffer)} tensors to {shard_name}")

        # Clear buffer
        self.buffer.clear()
        self.buffer_size = 0
        self.shard_counter += 1

        return shard_name

    def resume_manifest(self) -> Dict[str, Any]:
        """Snapshot of saver state for checkpoint (call after flush, buffer empty)."""
        if self.buffer:
            raise RuntimeError(
                "[HFSaver] resume manifest requires an empty buffer; call flush() first"
            )
        return {
            "shard_counter": self.shard_counter,
            "written_shards": sorted(self._written_shards),
            "weight_map": dict(self.weight_map),
            "tensor_type_map": dict(self.tensor_type_map),
        }

    @staticmethod
    def _make_shard_matcher(pattern: str) -> "re.Pattern[str]":
        """Build a regex that extracts the numeric shard index from a shard filename."""
        parts: list[str] = []
        has_field = False
        for literal, field_name, _spec, _conv in Formatter().parse(pattern):
            parts.append(re.escape(literal))
            if field_name is not None:
                has_field = True
                parts.append(r"(\d+)")
        if not has_field:
            raise ValueError(
                f"shard_name_pattern must contain one numeric format field: {pattern}"
            )
        return re.compile("^" + "".join(parts) + "$")

    def recover_from_disk(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Restore saver state from a checkpoint manifest of an interrupted run.

        - Restores shard_counter / weight_map / tensor_type_map / written_shards;
        - Deletes leftover *.tmp files and orphan shards flushed after the last
          checkpoint commit (not referenced by the manifest);
        - Validates every shard referenced by weight_map exists on disk.

        Must be called before adding any tensor. Returns a summary dict.
        """
        if self.buffer:
            raise RuntimeError(
                "[HFSaver] recover_from_disk must be called before adding tensors"
            )

        matcher = self._make_shard_matcher(self.shard_name_pattern)
        disk_shards: Dict[int, str] = {}
        for entry in sorted(self.output_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.name.endswith(".tmp"):
                # Leftover of an interrupted atomic write.
                entry.unlink()
                logger.warning(f"[HFSaver] Removed leftover temp file: {entry.name}")
                continue
            m = matcher.match(entry.name)
            if m:
                disk_shards[int(m.group(1))] = entry.name

        self.shard_counter = int(manifest.get("shard_counter", 0))
        self._written_shards = set(manifest.get("written_shards", []))
        self.weight_map = dict(manifest.get("weight_map", {}))
        self.tensor_type_map = dict(manifest.get("tensor_type_map", {}))

        # Orphan shards: on disk but flushed after the last checkpoint commit.
        orphans = [name for name in disk_shards.values() if name not in self._written_shards]
        for name in orphans:
            (self.output_dir / name).unlink()
        if orphans:
            preview = ", ".join(sorted(orphans)[:4])
            logger.warning(
                f"[HFSaver] Removed {len(orphans)} orphan shard(s) flushed after last "
                f"checkpoint: {preview}"
            )

        missing = sorted(
            {
                shard
                for shard in self.weight_map.values()
                if shard and not (self.output_dir / shard).exists()
            }
        )
        if missing:
            preview = ", ".join(missing[:4])
            raise IOError(f"[HFSaver] Resume manifest references missing shard(s): {preview}")

        # Keep counter beyond every written shard index (guards stale manifests).
        for name in self._written_shards:
            m = matcher.match(name)
            if m:
                self.shard_counter = max(self.shard_counter, int(m.group(1)) + 1)

        if bh.has_npu and self.require_tensor_types_on_npu:
            untyped = [k for k in self.weight_map if k not in self.tensor_type_map]
            if untyped:
                raise ValueError(
                    f"[HFSaver] Resumed weight_map has {len(untyped)} tensor(s) without "
                    "tensor_type entries; NPU output requires a manifest with "
                    "tensor_type_map"
                )

        return {
            "shards": len(self._written_shards),
            "tensors": len(self.weight_map),
            "orphan_shards_removed": len(orphans),
        }

    def _build_index(self) -> Dict[str, Any]:
        indexed_shards = {shard for shard in self.weight_map.values()}
        total_size = sum(
            (self.output_dir / shard).stat().st_size
            for shard in indexed_shards
            if (self.output_dir / shard).exists()
        )
        return {
            "metadata": {"total_size": int(total_size)},
            "weight_map": dict(sorted(self.weight_map.items())),
        }

    def _save_hf_assets(self) -> None:
        assets = (
            ("model config", self._model_config),
            ("tokenizer", self._tokenizer),
            ("processor", self._processor),
        )
        for label, obj in assets:
            if obj is None or not hasattr(obj, "save_pretrained"):
                continue
            try:
                obj.save_pretrained(self.output_dir)
            except Exception as exc:
                logger.warning(f"[HFSaver] Failed to save {label}: {exc}")

    def _build_ascend_quant_model_description(self, ascend_config: Dict[str, Any]) -> Dict[str, Any]:
        quant_type = str(ascend_config.get("model_quant_type", "FLOAT"))
        group_size = int(ascend_config.get("group_size", -1))

        description: Dict[str, Any] = {
            "version": "1.0.0",
            "model_quant_type": quant_type,
        }
        # Per-channel (group_size <= 0) must be explicitly written as 0,
        # otherwise vLLM defaults to 256 (per-group) and mis-decodes the weights.
        description["group_size"] = max(group_size, 0)

        missing_types = sorted(
            tensor_name
            for tensor_name in self.weight_map.keys()
            if tensor_name not in self.tensor_type_map
        )
        if missing_types and self.require_tensor_types_on_npu:
            preview = ", ".join(missing_types[:8])
            if len(missing_types) > 8:
                preview += ", ..."
            raise ValueError(
                "[HFSaver] Missing tensor types for Ascend quant description. "
                f"Examples: {preview}"
            )

        for tensor_name in sorted(self.weight_map.keys()):
            description[tensor_name] = self.tensor_type_map.get(tensor_name, "FLOAT")
        return description

    def _save_ascend_quant_description_if_needed(self) -> None:
        if self._model_config is None:
            return

        ascend_config = getattr(self._model_config, "ascend_quant_config", None)
        if not isinstance(ascend_config, dict):
            return

        description = self._build_ascend_quant_model_description(ascend_config)
        desc_path = self.output_dir / "quant_model_description.json"
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(description, f, indent=2)
        logger.info("[HFSaver] Wrote quant_model_description.json for Ascend runtime")

        if self.strip_quantization_config_on_npu:
            config_path = self.output_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    if "quantization_config" in config_data:
                        del config_data["quantization_config"]
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(config_data, f, indent=2)
                        logger.info("[HFSaver] Stripped quantization_config for Ascend output")
                except Exception as exc:
                    logger.warning(f"[HFSaver] Failed stripping quantization_config: {exc}")

    @staticmethod
    def _is_hidden_path(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)

    def _should_skip_source_file(self, rel_path: Path) -> bool:
        if self._is_hidden_path(rel_path):
            return True

        lower_name = rel_path.name.lower()
        if lower_name in self._SKIP_SOURCE_FILE_NAMES:
            return True

        if lower_name.startswith("model-") and lower_name.endswith(".safetensors"):
            return True
        if lower_name.startswith("pytorch_model"):
            return True

        if lower_name.endswith(self._WEIGHT_FILE_SUFFIXES):
            return True
        return False

    def _resolve_source_dir(self) -> Optional[Path]:
        if self._source_dir is not None and self._source_dir.exists():
            return self._source_dir

        if not self._source_ref:
            return None

        source_path = Path(self._source_ref)
        if source_path.exists():
            self._source_dir = source_path
            return source_path

        if self._source_model_hub != "hf":
            return None

        try:
            from huggingface_hub import snapshot_download

            revision = self._source_model_kwargs.get("revision")
            snapshot_dir = snapshot_download(
                repo_id=self._source_ref,
                revision=revision,
                ignore_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                    "*.onnx",
                    "*.msgpack",
                ],
            )
            self._source_dir = Path(snapshot_dir)
            return self._source_dir
        except Exception as exc:
            logger.warning(
                f"[HFSaver] Failed to download source snapshot for auxiliary files "
                f"('{self._source_ref}'): {exc}"
            )
            return None

    def _copy_aux_files_from_source(self) -> None:
        if not self.copy_aux_files:
            return

        source_dir = self._resolve_source_dir()
        if source_dir is None or not source_dir.exists():
            return

        copied = 0
        for source_file in source_dir.rglob("*"):
            if not source_file.is_file():
                continue
            rel_path = source_file.relative_to(source_dir)
            if self._should_skip_source_file(rel_path):
                continue

            target_file = self.output_dir / rel_path
            if target_file.exists():
                continue

            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied += 1

        if copied > 0:
            logger.info(f"[HFSaver] Copied {copied} auxiliary files from source model")

    def finalize(self) -> None:
        """Flush remaining buffer and write index plus HF auxiliary files."""
        self.flush()

        index = self._build_index()
        index_path = self.output_dir / "model.safetensors.index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        self._save_hf_assets()
        self._copy_aux_files_from_source()
        self._save_ascend_quant_description_if_needed()

        logger.success(
            f"[HFSaver] Finalized: tensors={len(self.weight_map)}, "
            f"shards={len(self._written_shards)}, output={self.output_dir}"
        )
