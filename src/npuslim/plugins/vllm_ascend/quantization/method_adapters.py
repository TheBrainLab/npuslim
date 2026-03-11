"""Patches for vllm_ascend/quantization/method_adapters.py

This module patches AscendLinearMethod to:
1. Support custom _input_dim/_output_dim from weight_dict
2. Fix the per-group parameter dimension bug in original vllm-ascend
"""

import torch
from vllm.model_executor.layers.linear import set_weight_attrs

from npuslim.plugins.registry import register_patch


@register_patch("vllm_ascend.quantization.method_adapters")
def patch_process_weight(module):
    """Patch AscendLinearMethod.process_weight to support custom dimensions.

    Original code hardcodes input_dim=1, output_dim=0.
    This patch allows reading from weight_dict via _input_dim/_output_dim keys.
    """

    def patched_process_weight(self, layer, weight_dict, extra_weight_attrs=None):
        # Extract packing information (if present)
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)
        # Extract custom dimension attributes (if present)
        custom_input_dim = weight_dict.pop("_input_dim", None)
        custom_output_dim = weight_dict.pop("_output_dim", None)

        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            # Use custom dimensions if provided, otherwise use defaults
            input_dim = custom_input_dim if custom_input_dim is not None else 1
            output_dim = custom_output_dim if custom_output_dim is not None else 0
            set_weight_attrs(param, {"input_dim": input_dim, "output_dim": output_dim})

            # Set packing attributes if the weight is packed
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(
                    param,
                    {
                        "packed_dim": packed_dim,
                        "packed_factor": packed_factor,
                    },
                )

            layer.register_parameter(weight_name, param)
            if extra_weight_attrs is not None:
                set_weight_attrs(param, extra_weight_attrs)

    module.AscendLinearMethod.process_weight = patched_process_weight


@register_patch("vllm_ascend.quantization.method_adapters")
def patch_process_pergroup_param(module):
    """Fix per-group parameter dimension bug.

    Original vllm-ascend has a bug:
    - Only sets output_dim=0
    - input_dim only set conditionally with duplicate assignment

    This patch always sets both dimensions correctly.
    """

    def patched_process_pergroup_param(self, layer, pergroup_dict, extra_weight_attrs=None):
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            # FIX: Always set both dimensions for per-group params
            # - output_dim=0: dimension 0 is output_size (for ColumnParallel sharding)
            # - input_dim=1: dimension 1 is num_groups (depends on input_size for RowParallel)
            set_weight_attrs(param, {"output_dim": 0, "input_dim": 1})
            layer.register_parameter(pergroup_name, param)
            if extra_weight_attrs is not None:
                set_weight_attrs(param, extra_weight_attrs)

    module.AscendLinearMethod.process_pergroup_param = patched_process_pergroup_param
