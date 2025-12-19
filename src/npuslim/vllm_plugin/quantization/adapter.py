import os
import json
from typing import Any, Dict, List, Optional

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


from ..utils import NPUSLIM_QUANTIZATION_METHOD


@register_quantization_config(NPUSLIM_QUANTIZATION_METHOD)
class NpuSlimConfig(QuantizationConfig):
    def __init__(self, quant_config: Dict[str, Any]):
        super().__init__()
        self.quant_description = quant_config

    @classmethod
    def get_name(cls) -> str:
        return NPUSLIM_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'Ascend hardware dose not support "get_min_capability" feature.'
        )

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NpuSlimConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return NPUSLIM_QUANTIZATION_METHOD
        return None

    def get_quant_method(self, layer, prefix) -> Optional["QuantizeMethodBase"]:
        # TODO
        ...

    def get_scaled_act_names(self) -> List[str]:
        return []
