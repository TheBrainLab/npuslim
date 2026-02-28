"""
HuggingFace transformers plugin for NPUSlim.

Provides automatic registration of NPUSlim quantization methods
with HuggingFace's quantization system.

Usage:
    # Import to trigger registration (via decorators in quip.py)
    import npuslim.plugins.transformers

    # Then load your model normally
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("path/to/quantized/model")
"""


def register():
    """
    Triggers registration of NPUSlim quantizers.
    Actually, the registration happens during module import via decorators,
    but we provide this function for consistency with the plugin system.
    """
    pass


from .quip import QuipConfig, QuipHfQuantizer

__all__ = ["QuipConfig", "QuipHfQuantizer"]
