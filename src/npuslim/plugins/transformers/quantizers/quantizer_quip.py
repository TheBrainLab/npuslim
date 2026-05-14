"""
QuIP quantization plugin for HuggingFace transformers.

Provides automatic loading of QuIP-quantized models via HF's
quantization_config mechanism.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

from transformers.quantizers import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from npuslim.plugins.registry import package_version_range, register_patch


@register_patch(
    registrar=register_quantization_config("quip"),
    condition=package_version_range("transformers", max_version="4.58.0"),
)
@dataclass
class QuipConfig(QuantizationConfigMixin):
    """
    QuIP quantization configuration.

    This config is stored in the model's config.json and used by
    HuggingFace to select the appropriate quantizer.
    """

    bits: int = 4
    quant_func: str = "rms"  # "rms" or "minmax"
    preproc_proj_mode: int = 2  # Butterfly mode
    checkpoint_format: str = "quip"

    def __post_init__(self):
        # Set quant_method as string - HF accepts both enum and string
        self.quant_method = "quip"

    def to_dict(self) -> dict:
        return {
            "quant_method": self.quant_method,
            "bits": self.bits,
            "quant_func": self.quant_func,
            "preproc_proj_mode": self.preproc_proj_mode,
            "checkpoint_format": self.checkpoint_format,
        }


@register_patch(
    registrar=register_quantizer("quip"),
    condition=package_version_range("transformers", max_version="4.58.0"),
)
class QuipHfQuantizer(HfQuantizer):
    """
    HuggingFace quantizer for QuIP.

    This class follows HF's HfQuantizer interface and is registered
    via the @register_quantizer decorator. When a model is loaded with
    quantization_config containing "quant_method": "quip", this quantizer
    is automatically used to process the model.
    """

    requires_calibration = False
    required_packages = ["safetensors"]

    def __init__(self, quantization_config: QuipConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        """Validate that required packages are available."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            raise ImportError(
                "safetensors is required for QuIP quantization. "
                "Install with: pip install safetensors"
            )

    def update_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """QuIP models use float16 by default."""
        return torch.float16

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs):
        """
        Replace nn.Linear with QuIPLinear BEFORE weights are loaded.

        This ensures that when HF loads the state_dict, the quantized weights
        (qweight, scales, etc.) have the correct target buffers to load into.
        """
        self._replace_with_quip_linear(model)
        return model

    def _process_model_after_weight_loading(self, model: nn.Module, **kwargs):
        """No post-processing needed - weights are loaded directly into QuIPLinear."""
        return model

    def _replace_with_quip_linear(self, model: nn.Module) -> int:
        """
        Replace nn.Linear layers with QuIPLinear.

        This is called BEFORE weight loading, so we use the original nn.Linear
        dimensions to create QuIPLinear layers. The weights will be loaded
        by HF's from_pretrained mechanism.

        Returns:
            Number of layers replaced
        """
        from npuslim.algorithms.quantization.quip.quip_algo import QuIPLinear

        replaced = 0
        bits = self.quantization_config.bits
        preproc_proj_mode = self.quantization_config.preproc_proj_mode

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if "layers" not in name:
                continue

            # Get parent module
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            # Get dimensions from original Linear
            infeatures = module.in_features
            outfeatures = module.out_features
            has_bias = module.bias is not None

            # Determine if zeros should be used based on quant_func
            has_zero = self.quantization_config.quant_func == "minmax"

            # Create QuIPLinear (don't move to device - HF will handle placement)
            quip_linear = QuIPLinear(
                bits=bits,
                infeatures=infeatures,
                outfeatures=outfeatures,
                has_zero=has_zero,
                bias=has_bias,
                proj_mode=preproc_proj_mode,
            )

            # Replace
            setattr(parent, child_name, quip_linear)
            replaced += 1

        return replaced

    @property
    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False
