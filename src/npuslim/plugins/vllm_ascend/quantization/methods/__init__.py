"""Quantization methods package."""
from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_linear import (
    AscendW4A16LinearMethod,
)
# from npuslim.plugins.vllm_ascend.quantization.methods.w4a16_moe import (
#     MoEWeightCollector,
#     fuse_expert_weights,
#     get_expert_layer_prefix,
#     get_mlp_layer_prefix,
#     is_moe_weight,
# )

__all__ = [
    "AscendW4A16LinearMethod",
    # "is_moe_weight",
    # "get_expert_layer_prefix",
    # "get_mlp_layer_prefix",
    # "fuse_expert_weights",
    # "MoEWeightCollector",
]
