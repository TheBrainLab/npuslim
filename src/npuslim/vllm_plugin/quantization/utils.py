from typing import Dict, Any
import importlib


class LazyMethodMap(dict):
    """一个只有在被 update 或访问时才真正加载模块的字典代理，以避免代码运行初期的循环导入问题"""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self._config = config
        self._cache = {}

    def _load_quant_type(self, quant_type: str) -> Dict[str, Any]:
        if quant_type not in self._cache:
            cfg = self._config[quant_type]
            module_path = cfg["module"]
            mod = importlib.import_module(module_path)

            methods = {}
            for key, class_name in cfg.items():
                if key == "module":
                    continue
                methods[key] = getattr(mod, class_name)

            self._cache[quant_type] = methods
        return self._cache[quant_type]

    def keys(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._load_quant_type(key)

    def __contains__(self, key):
        return key in self._config

    def __iter__(self):
        return iter(self._config)


NPUSLIM_QUANTIZATION_METHOD_MAP = LazyMethodMap(
    {
        "INT8Dynamic": {
            "module": "vllm_ascend.quantization.w8a8_dynamic",
            "linear": "AscendW8A8DynamicLinearMethod",
            "moe": "AscendW8A8DynamicFusedMoEMethod",
        },
    }
)