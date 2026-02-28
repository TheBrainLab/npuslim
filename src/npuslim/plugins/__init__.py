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
    from .vllm_ascend import register as register_vllm
    from .transformers import register as register_hf

    register_vllm()
    register_hf()


__all__ = ["register"]
