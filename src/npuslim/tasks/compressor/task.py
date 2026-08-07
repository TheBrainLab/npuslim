# src/npuslim/tasks/compressor/task.py
"""Compressor task for streaming quantization."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from npuslim.core.backend import bh
from npuslim.core import TaskRegistry
from npuslim.tasks.base_task import BaseTask
from npuslim.tasks.compressor.loader import ChunkLoader


@TaskRegistry.register("compressor", aliases=["CompressorTask", "QuantizeTask"])
class CompressorTask(BaseTask):
    """Streaming compression task."""

    def __init__(
        self,
        *,
        execution: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.execution_config = execution or {}

        # Execution settings
        self.mode = self.execution_config.get("mode", "full")
        self.chunk_size = max(int(self.execution_config.get("chunk_size", 1)), 1)

    def _create_loader(self) -> ChunkLoader:
        """Create chunk loader using model object."""
        model_kwargs = getattr(self._model_obj, "model_kwargs", {}) or {}
        model_device_map = model_kwargs.get("device_map")
        tensor_device = bh.resolve_device_map(
            model_device_map, default=bh.default_device_str()
        )

        block_name = getattr(
            self._model_obj,
            "block_name",
            getattr(self._model_obj, "layers_path", "model.layers"),
        )
        pre_module_names = list(
            getattr(self._model_obj, "pre_transformer_module_names", []) or []
        )
        post_module_names = list(
            getattr(self._model_obj, "post_transformer_module_names", ["lm_head"]) or ["lm_head"]
        )
        return ChunkLoader(
            model_path=getattr(self._model_obj, "path_str", str(getattr(self._model_obj, "path", ""))),
            model_hub=getattr(self._model_obj, "model_hub", "hf"),
            model_kwargs=model_kwargs,
            tensor_device=tensor_device,
            chunk_size=self.chunk_size,
            block_name=block_name,
            pre_module_names=pre_module_names,
            post_module_names=post_module_names,
            num_layers=getattr(self._model_obj, "num_transformer_layers", None),
        )

    @staticmethod
    def _build_skip_match_candidates(tensor_names: List[str]) -> List[str]:
        """
        Build candidate names for skip pattern matching from tensor keys.

        Includes:
        - full tensor names (e.g. model.layers.0.mlp.down_proj.weight)
        - leaf module names (e.g. model.layers.0.mlp.down_proj)
        """
        candidates: set[str] = set()
        for tensor_name in tensor_names:
            candidates.add(tensor_name)
            parts = tensor_name.split(".")
            if len(parts) > 1:
                candidates.add(".".join(parts[:-1]))
        return sorted(candidates)

    @staticmethod
    def _expand_patterns(all_names: List[str], patterns: List[str]) -> List[str]:
        """
        Expand direct/glob/regex patterns into concrete names.

        Regex patterns must be prefixed with `re:`.
        """
        expanded: set[str] = set()
        for pattern in patterns:
            if not pattern:
                continue

            matched: List[str] = []
            if pattern.startswith("re:"):
                regex_str = pattern[3:]
                try:
                    reg = re.compile(regex_str)
                except re.error as exc:
                    logger.error(f"[CompressorTask] Invalid regex pattern '{regex_str}': {exc}")
                    continue
                matched = [name for name in all_names if reg.fullmatch(name)]
            else:
                if pattern in all_names:
                    matched = [pattern]
                else:
                    matched = fnmatch.filter(all_names, pattern)

            if matched:
                expanded.update(matched)
            else:
                logger.debug(f"[CompressorTask] Skip pattern '{pattern}' matched nothing")
        return sorted(expanded)

    def _resolve_skip_layer_names(self, loader: ChunkLoader) -> List[str]:
        """
        Resolve skip patterns into concrete names for this model snapshot.
        """
        model_patterns = list(getattr(self._model_obj, "skip_layer_names", []) or [])
        user_patterns = list(self.params.get("ignore_layers", []) or [])
        combined_patterns = list(dict.fromkeys(model_patterns + user_patterns))
        if not combined_patterns:
            return []

        candidates = self._build_skip_match_candidates(loader.get_all_tensor_names())
        resolved = self._expand_patterns(candidates, combined_patterns)

        if resolved:
            display = "\n".join([f"    - {name}" for name in resolved[:5]])
            more = len(resolved) - 5
            if more > 0:
                display += f"\n    - ... and {more} more."
            logger.info(
                "[CompressorTask] Resolved skip names from patterns:\n"
                f"{display}"
            )
        return resolved

    def run(self) -> Dict[str, Any]:
        """Execute streaming compression."""
        if self.rm is None:
            raise ValueError("resource_manager is required")
        if self.model_ref is None:
            raise ValueError("model reference is required")
        if self._model_obj is None:
            raise ValueError("model not acquired")

        # Create components
        loader = self._create_loader()
        algo = self._algorithm
        saver = self._saver

        loader.refresh_index()
        if saver is not None:
            if hasattr(saver, "set_source"):
                saver.set_source(
                    loader.resolve_model_source(),
                    model_hub=getattr(loader, "model_hub", "hf"),
                    model_kwargs=getattr(loader, "model_kwargs", {}),
                )
            if hasattr(saver, "set_hf_assets"):
                saver.set_hf_assets(
                    model_config=getattr(self._model_obj, "config", None),
                    tokenizer=getattr(self._model_obj, "tokenizer", None),
                    processor=getattr(self._model_obj, "processor", None),
                )
        resolved_skip_layer_names = self._resolve_skip_layer_names(loader)
        if hasattr(algo, "set_runtime_context"):
            algo.set_runtime_context(
                model_obj=self._model_obj,
                model_config=getattr(self._model_obj, "config", None),
                skip_layer_names=resolved_skip_layer_names,
            )
        all_original_keys = set(loader.get_all_tensor_names())
        touched_original_keys: set[str] = set()
        has_any_tensors = loader.get_total_tensors() > 0
        if not has_any_tensors:
            raise ValueError(
                "[CompressorTask] Loader found 0 tensors. "
                "Model checkpoint is unsupported or missing. "
                "Expected one of: model.safetensors(.index.json), "
                "pytorch_model.bin(.index.json)."
            )
        if self.mode == "full":
            chunk_count = 1 if has_any_tensors else 0
        else:
            chunk_count = loader.get_chunk_count()

        logger.info(
            f"[CompressorTask] Starting: mode={self.mode}, "
            f"chunks={chunk_count}, chunk_size={self.chunk_size}, device={loader.tensor_device}"
        )

        output_dir = None
        algo.on_start()
        try:
            if self.mode == "full":
                if chunk_count == 0:
                    logger.warning("[CompressorTask] No tensors found for full load")
                else:
                    chunk = loader.load_full()
                    chunk.calib_data = self._calib_data
                    chunk.metadata["skip_layer_names"] = list(resolved_skip_layer_names)
                    touched_original_keys.update(chunk.all_tensors().keys())

                    chunk = algo.process_chunk(chunk)

                    if saver is not None:
                        saver.add_tensors(
                            chunk.all_tensors(),
                            tensor_types=self._resolve_chunk_tensor_types(chunk, npu_strict=bh.has_npu),
                        )

                    loader.unload_chunk(0)
            else:
                for chunk_idx in range(chunk_count):
                    chunk = loader.load_chunk(chunk_idx)
                    chunk.calib_data = self._calib_data
                    chunk.metadata["skip_layer_names"] = list(resolved_skip_layer_names)
                    touched_original_keys.update(chunk.all_tensors().keys())

                    chunk = algo.process_chunk(chunk)

                    if saver is not None:
                        saver.add_tensors(
                            chunk.all_tensors(),
                            tensor_types=self._resolve_chunk_tensor_types(chunk, npu_strict=bh.has_npu),
                        )

                    loader.unload_chunk(chunk_idx)

            # MTP layer quantization (after regular layers, before backfill)
            mtp_names = list(getattr(self._model_obj, "mtp_layer_names", []))
            quantize_mtp = getattr(algo, "_quantize_mtp", False)
            save_mtp_debug = getattr(algo, "_save_mtp_debug", False)
            # Only process MTP when explicitly requested (quantize or save debug).
            # Otherwise the MTP layer is left to the backfill path, which saves
            # the original per-expert 2D FLOAT checkpoint format unchanged.
            process_mtp = mtp_names and saver is not None and (quantize_mtp or save_mtp_debug)
            if process_mtp:
                if quantize_mtp:
                    logger.info(f"[CompressorTask] MTP quantization enabled for: {mtp_names}")
                else:
                    logger.info(f"[CompressorTask] MTP debug-only mode (save_mtp_debug=True, no quantization): {mtp_names}")
                mtp_chunk = self._load_mtp_chunk(loader, mtp_names, resolved_skip_layer_names)
                if mtp_chunk is not None:
                    touched_original_keys.update(mtp_chunk.all_tensors().keys())
                    mtp_chunk = algo.process_mtp_chunk(mtp_chunk)
                    saver.add_tensors(
                        mtp_chunk.all_tensors(),
                        tensor_types=self._resolve_chunk_tensor_types(mtp_chunk, npu_strict=bh.has_npu),
                    )
                    loader.unload_chunk(999)  # Release shard handles
            elif mtp_names and saver is not None:
                # MTP layers are not being quantized or debug-saved.
                # Do NOT mark them as touched -- let the backfill path below
                # load them from checkpoint and save as FLOAT, so they are
                # present in the output shards and index files.
                logger.info(
                    f"[CompressorTask] MTP layers {mtp_names} will be saved as-is "
                    f"(per-expert 2D FLOAT, quantize_mtp=False, save_mtp_debug=False) "
                    f"via backfill"
                )

            if saver is not None:
                missing_original_keys = sorted(all_original_keys - touched_original_keys)
                if missing_original_keys:
                    # Categorize missing keys by layer for better visibility
                    layer_counts: dict[str, int] = {}
                    for k in missing_original_keys:
                        parts = k.split(".")
                        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                            layer_key = f"layer {parts[2]}"
                        else:
                            layer_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                        layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1
                    layer_summary = ", ".join(f"{k}={v}" for k, v in sorted(layer_counts.items()))
                    logger.warning(
                        f"[CompressorTask] Backfilling {len(missing_original_keys)} untouched original tensors "
                        f"({layer_summary})"
                    )
                    logger.info(f"[CompressorTask] Backfill: loading {len(missing_original_keys)} tensors from checkpoint")
                    missing_tensors = loader.load_tensors(missing_original_keys)
                    logger.info(
                        f"[CompressorTask] Backfill: loaded {len(missing_tensors)} tensors, "
                        f"saving as FLOAT to output shards"
                    )
                    saver.add_tensors(
                        missing_tensors,
                        tensor_types={name: "FLOAT" for name in missing_original_keys},
                    )
                    logger.success(
                        f"[CompressorTask] Backfill: completed, {len(missing_tensors)} tensors saved as FLOAT"
                    )
                else:
                    logger.info("[CompressorTask] Backfill: no untouched tensors, nothing to backfill")
        finally:
            try:
                algo.on_finish()
            finally:
                if saver is not None:
                    saver.finalize()
                    output_dir = getattr(saver, "output_dir", None)
                    if output_dir:
                        logger.success(f"[CompressorTask] Saved to: {output_dir}")
                loader.close()

        logger.success("[CompressorTask] Completed")

        self.rm.publish_model_state(
            self.model_ref,
            self._model_obj,
            state_meta={
                "quantized": True,
                "algorithm": self.algorithm_config.get("type"),
                "output_dir": str(output_dir) if output_dir else None,
            },
        )

        return {
            "chunks_processed": chunk_count,
            "output_dir": str(output_dir) if output_dir else None,
        }

    def _load_mtp_chunk(self, loader: ChunkLoader, mtp_names: List[str], skip_layer_names: List[str]):
        """Load MTP layer tensors from checkpoint into a ChunkContext."""
        from npuslim.tasks.compressor.context import ChunkContext, LayerInfo

        all_tensor_names = loader.get_all_tensor_names()
        layers: List[LayerInfo] = []

        for mtp_name in mtp_names:
            # Find tensors belonging to this MTP layer
            mtp_tensor_names = [t for t in all_tensor_names if t.startswith(f"{mtp_name}.")]
            if not mtp_tensor_names:
                logger.warning(f"[CompressorTask] No tensors found for MTP layer: {mtp_name}")
                continue

            mtp_tensors = loader.load_tensors(mtp_tensor_names)

            # Convert to relative names (strip layer prefix)
            layer_prefix = f"{mtp_name}."
            layer_tensors = {}
            for full_name, tensor in mtp_tensors.items():
                rel_name = full_name[len(layer_prefix):] if full_name.startswith(layer_prefix) else full_name
                layer_tensors[rel_name] = tensor

            layer_idx = int(mtp_name.split(".")[-1])
            layers.append(LayerInfo(name=mtp_name, index=layer_idx, tensors=layer_tensors))
            logger.info(f"[CompressorTask] Loaded MTP layer: {mtp_name} ({len(layer_tensors)} tensors)")

        if not layers:
            return None

        chunk = ChunkContext(
            chunk_index=999,
            layers=layers,
            pre_modules=[],
            post_modules=[],
        )
        chunk.calib_data = self._calib_data
        chunk.metadata["skip_layer_names"] = list(skip_layer_names)
        return chunk

    @staticmethod
    def _resolve_chunk_tensor_types(
        chunk,
        *,
        npu_strict: bool,
    ) -> Optional[Dict[str, str]]:
        all_names = list(chunk.all_tensors().keys())
        tensor_types = chunk.metadata.get("tensor_types")

        if tensor_types is None:
            if npu_strict:
                raise ValueError(
                    "[CompressorTask] NPU mode requires chunk.metadata['tensor_types'] "
                    "from algorithm for every tensor."
                )
            return None

        if not isinstance(tensor_types, dict):
            raise ValueError(
                "[CompressorTask] chunk.metadata['tensor_types'] must be a dict[str, str]"
            )

        resolved: Dict[str, str] = {}
        missing: List[str] = []
        for name in all_names:
            t = tensor_types.get(name)
            if t is None:
                if npu_strict:
                    missing.append(name)
                else:
                    resolved[name] = "FLOAT"
            else:
                resolved[name] = str(t)

        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            raise ValueError(
                "[CompressorTask] Missing tensor types for NPU chunk save. "
                f"Examples: {preview}"
            )
        return resolved
