# npuslim/vllm_plugin.py
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

def register():
    register_quant()

def register_quant():
    try:
        from vllm_ascend.quantization.utils import ASCEND_QUANTIZATION_METHOD_MAP
        from .quantization.utils import NPUSLIM_QUANTIZATION_METHOD_MAP
        
        ASCEND_QUANTIZATION_METHOD_MAP.update(NPUSLIM_QUANTIZATION_METHOD_MAP)
        
    except ImportError:
        pass