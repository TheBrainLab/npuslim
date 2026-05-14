"""NPUSlim vLLM Core Plugin.

Patches vLLM core modules (e.g., model_executor/models) for NPUSlim compatibility.
"""

from pathlib import Path

from npuslim.plugins.logging import patch_logger


def register():
    """Register NPUSlim extensions with vLLM core.

    This function discovers and applies patches to vLLM core modules.
    """
    try:
        from npuslim.plugins.registry import apply_all_patches, discover_modules
        # Discover vllm plugin modules (model_executor, etc.)
        plugin_dir = str(Path(__file__).parent)
        discover_modules("npuslim.plugins.vllm", plugin_dir)
        # Apply all registered patches
        apply_all_patches()
        patch_logger.info("Registered NPUSlim with vLLM core")
    except ImportError as e:
        patch_logger.warning(f"Could not register NPUSlim with vLLM core: {e}")
