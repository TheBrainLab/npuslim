# npuslim/vllm_plugin.py
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

def register():
    register_quant()
    # try:
    #     from vllm_ascend.quantization.quant_config import AscendQuantConfig
    #     from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
    #     from vllm.model_executor.layers.linear import LinearBase
    
    #     original_get_quant_method = AscendQuantConfig.get_quant_method

    #     def patched_get_quant_method(self, layer: torch.nn.Module, prefix: str):
    #         quant_type = self.quant_description.get('quant_type')
            
    #         if quant_type == 'INT8Dynamic' and isinstance(layer, LinearBase):
    #             logger.info(f"NpuSlim Plugin: Hooking {prefix} with INT8 Dynamic")
    #             return AscendW8A8DynamicLinearMethod(self, prefix, self.packed_modules_mapping)
            
    #         return original_get_quant_method(self, layer, prefix)

    #     AscendQuantConfig.get_quant_method = patched_get_quant_method
    #     logger.info("NpuSlim Plugin: Successfully patched vllm-ascend")

    # except ImportError:
    #     logger.warning("NpuSlim Plugin: vllm-ascend not found, skipping patch.")
    # except Exception as e:
    #     logger.error(f"NpuSlim Plugin: Failed to apply patch: {e}")


def register_quant():
    try:
        from vllm_ascend.quantization.utils import ASCEND_QUANTIZATION_METHOD_MAP
        from .quantization.utils import NPUSLIM_QUANTIZATION_METHOD_MAP
        
        # 批量注入所有支持的算法，此时不会触发任何 import
        # 只有当 vllm 匹配到某个 quant_type 并从这个 map 里取值时，才会真的 import
        ASCEND_QUANTIZATION_METHOD_MAP.update(NPUSLIM_QUANTIZATION_METHOD_MAP)
        
    except ImportError:
        pass