"""
Sparse24 quantization plugin for HuggingFace transformers.

Provides automatic loading of 2:4 structured sparse models via HF's
quantization_config mechanism. Replaces nn.Linear with AscendSparse24Linear
so that packed sparse tensors (weight, weight_scale, weight_index) are loaded
directly from the safetensors checkpoint.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.quantizers import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from npuslim.plugins.registry import always_disable, register_patch


# NOTE: Transformers does not support Ascend NPU sparse inference yet.
# Disable registration until the framework adds native support.
@register_patch(
    registrar=register_quantization_config("sparse24"),
    condition=always_disable,
)
@dataclass
class Sparse24Config(QuantizationConfigMixin):
    """Sparse24 quantization configuration stored in config.json."""

    sparsity_type: str = "2:4"

    def __post_init__(self):
        self.quant_method = "sparse24"

    def to_dict(self) -> dict:
        return {
            "quant_method": self.quant_method,
            "sparsity_type": self.sparsity_type,
        }


@register_patch(
    registrar=register_quantizer("sparse24"),
    condition=always_disable,
)
class Sparse24HfQuantizer(HfQuantizer):
    """HuggingFace quantizer for 2:4 structured sparse models.

    Replaces nn.Linear layers with AscendSparse24Linear before weight loading
    so that packed sparse tensors are loaded into the correct buffers.
    """

    requires_calibration = False
    required_packages = ["safetensors"]

    def __init__(self, quantization_config: Sparse24Config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        try:
            import safetensors  # noqa: F401
        except ImportError:
            raise ImportError(
                "safetensors is required for Sparse24. "
                "Install with: pip install safetensors"
            )

    def update_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return torch.float16

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs):
        self._replace_with_sparse24_linear(model)
        return model

    def _process_model_after_weight_loading(self, model: nn.Module, **kwargs):
        return model

    def _replace_with_sparse24_linear(self, model: nn.Module) -> int:
        from npuslim.algorithms.quantization.sparsegpt.sparsegpt_algo import (
            AscendSparse24Linear,
        )

        replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if "layers" not in name:
                continue

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            sparse_linear = AscendSparse24Linear(
                infeatures=module.in_features,
                outfeatures=module.out_features,
                bias=module.bias is not None,
            )
            setattr(parent, child_name, sparse_linear)
            replaced += 1

        return replaced

    @property
    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False
