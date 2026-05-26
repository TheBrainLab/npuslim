"""Common runtime helpers for quantization algorithms."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Iterable, List, Optional

from npuslim.algorithms.base_algo import BaseAlgorithm
from npuslim.core.backend import bh


class BaseQuantizationAlgorithm(BaseAlgorithm):
    """Shared runtime context and skip-matching utilities."""

    _TAG: str = "Base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_obj: Optional[Any] = None
        self._model_config: Optional[Any] = None
        self._skip_layer_names: List[str] = []
        self._save_backend: Optional[str] = None

    def set_runtime_context(
        self,
        *,
        model_obj: Any = None,
        model_config: Any = None,
        skip_layer_names: Optional[List[str]] = None,
    ) -> None:
        self._model_obj = model_obj
        self._model_config = model_config
        if skip_layer_names is not None:
            self._skip_layer_names = list(skip_layer_names)

    @property
    def target_backend(self) -> str:
        """Backend governing the output packing format.

        Falls back to the runtime backend when ``_save_backend`` is not set.
        """
        return self._save_backend or bh.detected_name

    @staticmethod
    def should_skip_name(full_name: str, skip_layer_names: Iterable[str]) -> bool:
        for skip_name in skip_layer_names:
            if not skip_name:
                continue

            if skip_name.startswith("re:"):
                try:
                    if re.fullmatch(skip_name[3:], full_name):
                        return True
                except re.error:
                    continue
                continue

            if full_name == skip_name or full_name.startswith(f"{skip_name}."):
                return True

            if fnmatch.fnmatch(full_name, skip_name):
                return True
        return False

    def _set_skip_from_chunk_metadata(self, chunk) -> List[str]:
        skip_layer_names = list(chunk.metadata.get("skip_layer_names", []) or [])
        if skip_layer_names:
            self._skip_layer_names = skip_layer_names
        return skip_layer_names

    def _mark_model_quantized(self) -> None:
        if self._model_obj is not None and hasattr(self._model_obj, "quantized"):
            self._model_obj.quantized = True
