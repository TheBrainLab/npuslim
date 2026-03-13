"""
NPUSlim Plugin System.

Provides automatic registration of NPUSlim quantization methods
with various deployment backends (vLLM, HuggingFace, etc.).
"""


def register():
    """
    Register all NPUSlim plugins with their respective frameworks.

    Call this once after installing npuslim, or it happens automatically
    via entry points.
    """
    from .vllm import register as register_vllm_core
    from .vllm_ascend import register as register_vllm_ascend
    from .transformers import register as register_hf

    # Register vLLM core patches first (model patches, etc.)
    register_vllm_core()
    # Then register vLLM-Ascend specific patches
    register_vllm_ascend()
    # Finally register HuggingFace transformers patches
    register_hf()


__all__ = ["register"]
