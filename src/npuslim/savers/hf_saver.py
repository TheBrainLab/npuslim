"""HuggingFace-format streaming saver."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger
from safetensors.torch import save_file

from npuslim.registry import SaverRegistry
from npuslim.savers.base_saver import BaseSaver


@SaverRegistry.register("HuggingFaceSaver", aliases=["HF"])
class HuggingFaceSaver(BaseSaver):
    """Streaming safetensors saver with HuggingFace-compatible output layout."""

    _WEIGHT_FILE_SUFFIXES = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".msgpack",
    )
    _SKIP_SOURCE_FILE_NAMES = {
        "model.safetensors.index.json",
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
        output_dir: Path | str | None = None,
        save_dir: Path | str | None = None,
        size_threshold: int | str = 4 * 1024 * 1024 * 1024,  # 4 GiB
        max_shard_size: int | str | None = None,
        shard_size: int | str | None = None,
        shard_name_pattern: str = "model-{:05d}.safetensors",
        copy_aux_files: bool = True,
    ):
        if output_dir is None:
            output_dir = save_dir
        if output_dir is None:
            raise ValueError("HuggingFaceSaver requires 'output_dir' (or legacy 'save_dir').")

        threshold_source = max_shard_size if max_shard_size is not None else shard_size
        if threshold_source is None:
            threshold_source = size_threshold

        self.output_dir = Path(output_dir)
        self.size_threshold = self._parse_size_to_bytes(threshold_source)
        self.shard_name_pattern = shard_name_pattern
        self.copy_aux_files = bool(copy_aux_files)

        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_size: int = 0
        self.shard_counter: int = 0
        self.weight_map: Dict[str, str] = {}
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

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add tensor to buffer, auto-flush if threshold exceeded."""
        if name in self.buffer:
            old_tensor = self.buffer[name]
            self.buffer_size -= old_tensor.numel() * old_tensor.element_size()

        tensor_size = tensor.numel() * tensor.element_size()

        # Flush if adding would exceed threshold
        if self.buffer_size + tensor_size > self.size_threshold and self.buffer:
            self.flush()

        self.buffer[name] = tensor.cpu().contiguous()
        self.buffer_size += tensor_size

        # Flush immediately once threshold is reached.
        if self.buffer_size >= self.size_threshold:
            self.flush()

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

        logger.success(
            f"[HFSaver] Finalized: tensors={len(self.weight_map)}, "
            f"shards={len(self._written_shards)}, output={self.output_dir}"
        )
