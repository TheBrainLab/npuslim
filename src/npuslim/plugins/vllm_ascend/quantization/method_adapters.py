"""Patches for vllm_ascend/quantization/method_adapters.py

This module patches AscendLinearMethod.create_weights to fix the per-group
parameter dimension bug in original vllm-ascend and to propagate quant-method
parameter attributes onto created Parameters.

Bug in upstream (vllm_ascend/quantization/method_adapters.py:105-116):
```python
for pergroup_name, pergroup_param in pergroup_dict.items():
    param = torch.nn.Parameter(pergroup_param, requires_grad=False)
    set_weight_attrs(param, {"output_dim": 0})
    layer.register_parameter(pergroup_name, param)
    set_weight_attrs(param, extra_weight_attrs)
    if (
        "weight_scale_second" in pergroup_name
        or "weight_offset_second" in pergroup_name
        or is_mx_quant_type(self.quant_method)
    ):
        param.input_dim = 1
        param.input_dim = 1  # Duplicate line (minor bug)
```

Problem: Only sets input_dim=1 for *_second params, missing for regular
weight_scale/weight_offset in RowParallelLinear layers.

For RowParallelLinear (o_proj, down_proj), per-group params depend on input_size,
so input_dim=1 must be set for proper tensor parallel sharding.
"""

from vllm.model_executor.layers.linear import RowParallelLinear

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch


@register_patch(
    target="vllm_ascend.quantization.method_adapters",
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
def patch_create_weights(module):
    """Patch AscendLinearMethod.create_weights to fix per-group param dimensions.

    This patch wraps the original create_weights and fixes the input_dim attribute
    for per-group parameters in RowParallelLinear layers.
    """

    original_create_weights = module.AscendLinearMethod.create_weights

    def patched_create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Call original implementation
        original_create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        # FIX: For RowParallelLinear, ensure per-group params have input_dim=1
        # The original code only sets this for *_second params, but regular
        # weight_scale/weight_offset also need it for proper TP sharding.
        if isinstance(layer, RowParallelLinear):
            for name, param in layer.named_parameters(recurse=False):
                if name in ("weight_scale", "weight_offset"):
                    if (
                        param.ndim > 1
                        and param.shape[1] > 1
                        and (not hasattr(param, "input_dim") or param.input_dim is None)
                    ):
                        param.input_dim = 1

        get_param_extra_attrs = getattr(
            self.quant_method, "get_param_extra_attrs", None
        )
        if callable(get_param_extra_attrs):
            for name, param in layer.named_parameters(recurse=False):
                extra_attrs = get_param_extra_attrs(name)
                if extra_attrs:
                    module.set_weight_attrs(param, extra_attrs)

    module.AscendLinearMethod.create_weights = patched_create_weights
    patch_logger.info(
        "Patched AscendLinearMethod.create_weights to support per-group "
        "input_dim parameters and quant-method parameter attrs"
    )
