"""NPUSlim vLLM-Ascend Plugin.

Registers NPUSlim quantization methods with vLLM-Ascend for NPU deployment.
"""

from pathlib import Path

from npuslim.plugins.logging import patch_logger


def register():
    """Register NPUSlim extensions with vLLM-Ascend.

    This function discovers and applies patches to vLLM-Ascend modules.
    For vLLM core patches (e.g., model patches), see npuslim.plugins.vllm.

    Schemes use vllm-ascend's @register_scheme decorator.
    Patches use our @register_patch decorator from npuslim.plugins.registry.
    """
    try:
        from npuslim.plugins.registry import apply_all_patches, discover_modules

        # Discover vllm_ascend plugin modules
        # This triggers @register_patch and @register_scheme decorators
        plugin_dir = str(Path(__file__).parent)
        discover_modules("npuslim.plugins.vllm_ascend", plugin_dir)

        # Apply all registered patches
        apply_all_patches()

        patch_logger.info("Registered NPUSlim with vLLM-Ascend")

    except ImportError as e:
        patch_logger.warning(f"Could not register NPUSlim with vLLM-Ascend: {e}")
