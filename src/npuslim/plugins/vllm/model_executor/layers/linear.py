"""Patches for vllm/model_executor/layers/linear.py.

Adds opt-in stacked weight loader dispatch for parameters that need a custom
merge path when loaded into fused QKV / merged-column layers.
"""

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch
from npuslim.plugins.vllm.model_executor.layers._stacked_sparse24 import (
    STACKED_WEIGHT_LOADER_ATTR,
)


@register_patch(
    target="vllm.model_executor.layers.linear",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_stacked_weight_loader_dispatch(module):
    """Patch stacked linear loaders to honor opt-in custom shard merge hooks."""

    original_qkv_weight_loader = module.QKVParallelLinear.weight_loader
    original_merged_weight_loader = module.MergedColumnParallelLinear.weight_loader

    def patched_qkv_weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        special_loader = getattr(param, STACKED_WEIGHT_LOADER_ATTR, None)
        if special_loader is not None and loaded_shard_id is not None:
            special_loader(self, param, loaded_weight, loaded_shard_id)
            return
        return original_qkv_weight_loader(self, param, loaded_weight, loaded_shard_id)

    def patched_merged_weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        special_loader = getattr(param, STACKED_WEIGHT_LOADER_ATTR, None)
        if (
            special_loader is not None
            and loaded_shard_id is not None
            and not isinstance(loaded_shard_id, tuple)
        ):
            special_loader(self, param, loaded_weight, loaded_shard_id)
            return
        return original_merged_weight_loader(
            self, param, loaded_weight, loaded_shard_id
        )

    module.QKVParallelLinear.weight_loader = patched_qkv_weight_loader
    module.MergedColumnParallelLinear.weight_loader = patched_merged_weight_loader
    patch_logger.info(
        "Patched stacked linear weight loaders to honor opt-in custom merge hooks"
    )
