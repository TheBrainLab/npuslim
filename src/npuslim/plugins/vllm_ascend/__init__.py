"""NPUSlim vLLM-Ascend Plugin.

Registers NPUSlim quantization methods with vLLM-Ascend for NPU deployment.
"""

from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def register():
    """Register NPUSlim extensions with vLLM-Ascend.

    This function:
    1. Discovers all modules (patches and schemes self-register via decorators)
    2. Applies all registered patches to vllm-ascend modules

    Schemes use vllm-ascend's @register_scheme decorator.
    Patches use our @register_patch decorator from npuslim.plugins.registry.
    """
    try:
        from npuslim.plugins.registry import apply_all_patches, discover_modules

        # Discover all modules under this plugin directory
        # This triggers @register_patch and @register_scheme decorators
        plugin_dir = str(Path(__file__).parent)
        discover_modules("npuslim.plugins.vllm_ascend", plugin_dir)

        # Apply all registered patches
        apply_all_patches()

        logger.info("NPUSlim registered with vLLM-Ascend")

    except ImportError as e:
        logger.warning(f"Could not register NPUSlim with vLLM-Ascend: {e}")
