"""
NPUSlim Plugin System.

Provides automatic registration of NPUSlim quantization methods
with various deployment backends (vLLM, HuggingFace, etc.).
"""

_REGISTERED = False


def register():
    """
    Register all NPUSlim plugins with their respective frameworks.

    Call this once after installing npuslim, or it happens automatically
    via entry points. This function is idempotent - multiple calls are safe.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from npuslim.core.backend import bh

    from .transformers import register as register_hf
    from .vllm import register as register_vllm_core

    # Register vLLM core patches first (model patches, etc.)
    register_vllm_core()
    # Register HuggingFace transformers patches
    register_hf()

    # Only register vLLM-Ascend specific patches when NPU is available
    if bh.name == "npu":
        from .vllm_ascend import register as register_vllm_ascend

        register_vllm_ascend()

    _REGISTERED = True


__all__ = ["register"]
