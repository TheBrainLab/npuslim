"""NPUSlim Transformers Plugin.

Provides quantization extensions for HuggingFace Transformers.

Quantizers are registered via entry points in pyproject.toml:
    [project.entry-points."transformers.quantizers"]
    quip = "npuslim.plugins.transformers.quantizers.quantizer_quip:QuipHfQuantizer"

Transformers loads these on-demand when a quantized model is requested.
"""


def register():
    """Register NPUSlim extensions with Transformers.

    Note: Transformers quantizers are auto-registered via entry points
    when the model is loaded with a quantization config. This function
    is for any additional patches that need to be applied at startup.
    """
    # Currently no patches needed - quantizers use decorators
    pass


# Import for direct access
from .quantizers.quantizer_quip import QuipConfig, QuipHfQuantizer

__all__ = ["QuipConfig", "QuipHfQuantizer"]
