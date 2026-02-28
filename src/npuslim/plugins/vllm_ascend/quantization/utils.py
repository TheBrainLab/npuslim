"""
vLLM-Ascend quantization method registry for NPUSlim.

Provides lazy-loading quantization methods to avoid circular imports.
"""

from typing import Dict, Any
import importlib


class LazyMethodMap:
    """
    Lazy-loading dictionary for quantization methods.

    Defers module imports until first access to avoid circular import issues
    during vLLM-Ascend initialization.
    """

    def __init__(self, config: Dict[str, Dict[str, str]]):
        """
        Args:
            config: Mapping from quantization type to module and class names.
                Example: {
                    "INT8Dynamic": {
                        "module": "vllm_ascend.quantization.w8a8_dynamic",
                        "linear": "AscendW8A8DynamicLinearMethod",
                        "moe": "AscendW8A8DynamicFusedMoEMethod",
                    }
                }
        """
        self._config = config
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load(self, quant_type: str) -> Dict[str, Any]:
        """Load and cache methods for a quantization type."""
        if quant_type in self._cache:
            return self._cache[quant_type]

        cfg = self._config[quant_type]
        module_path = cfg["module"]
        mod = importlib.import_module(module_path)

        methods = {
            key: getattr(mod, class_name)
            for key, class_name in cfg.items()
            if key != "module"
        }
        self._cache[quant_type] = methods
        return methods

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self._load(key)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __iter__(self):
        return iter(self._config)

    def keys(self):
        return self._config.keys()

    def items(self):
        """Iterate with lazy loading."""
        for key in self._config:
            yield key, self[key]


# Registry of NPUSlim quantization methods for vLLM-Ascend.
# These methods are merged into vLLM-Ascend's ASCEND_QUANTIZATION_METHOD_MAP.
NPUSLIM_QUANTIZATION_METHOD_MAP = LazyMethodMap({
    "INT8Dynamic": {
        "module": "vllm_ascend.quantization.w8a8_dynamic",
        "linear": "AscendW8A8DynamicLinearMethod",
        "moe": "AscendW8A8DynamicFusedMoEMethod",
    },
})
