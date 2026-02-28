"""
NPUSlim vLLM-Ascend Plugin.

Registers NPUSlim quantization methods with vLLM-Ascend for NPU deployment.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def register():
    """
    Register NPUSlim quantization methods with vLLM-Ascend.

    This function is called automatically via entry point when vLLM starts.
    """
    try:
        from vllm_ascend.quantization.utils import ASCEND_QUANTIZATION_METHOD_MAP
    except ImportError:
        # vLLM-Ascend not installed
        logger.debug("vLLM-Ascend not available, skipping registration")
        return

    try:
        from .quantization.utils import NPUSLIM_QUANTIZATION_METHOD_MAP

        # Update the quantization method map
        for key, value in NPUSLIM_QUANTIZATION_METHOD_MAP.items():
            ASCEND_QUANTIZATION_METHOD_MAP[key] = value

        logger.info("NPUSlim quantization methods registered with vLLM-Ascend")

    except ImportError as e:
        logger.warning(f"Could not load NPUSlim quantization methods: {e}")