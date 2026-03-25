# src/npuslim/tasks/compressor/task.py
"""Compressor task for streaming quantization."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from npuslim.core.backend import bh
from npuslim.registry import TaskRegistry
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
        self.device = self.execution_config.get("device", bh.default_device_str())
        self.strict_assignment = bool(self.execution_config.get("strict_assignment", True))

    def _create_loader(self) -> ChunkLoader:
        """Create chunk loader using model object."""
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
            model_kwargs=getattr(self._model_obj, "model_kwargs", {}),
            tensor_device=self.device,
            chunk_size=self.chunk_size,
            block_name=block_name,
            pre_module_names=pre_module_names,
            post_module_names=post_module_names,
            strict_assignment=self.strict_assignment,
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
        if self.mode == "full":
            chunk_count = 1 if has_any_tensors else 0
        else:
            chunk_count = loader.get_chunk_count()

        logger.info(
            f"[CompressorTask] Starting: mode={self.mode}, "
            f"chunks={chunk_count}, chunk_size={self.chunk_size}, device={self.device}"
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
                        saver.add_tensors(chunk.all_tensors())

                    loader.unload_chunk(0)
            else:
                for chunk_idx in range(chunk_count):
                    chunk = loader.load_chunk(chunk_idx)
                    chunk.calib_data = self._calib_data
                    chunk.metadata["skip_layer_names"] = list(resolved_skip_layer_names)
                    touched_original_keys.update(chunk.all_tensors().keys())

                    chunk = algo.process_chunk(chunk)

                    if saver is not None:
                        saver.add_tensors(chunk.all_tensors())

                    loader.unload_chunk(chunk_idx)

            if saver is not None:
                missing_original_keys = sorted(all_original_keys - touched_original_keys)
                if missing_original_keys:
                    logger.warning(
                        f"[CompressorTask] Backfilling {len(missing_original_keys)} untouched original tensors"
                    )
                    missing_tensors = loader.load_tensors(missing_original_keys)
                    saver.add_tensors(missing_tensors)
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
