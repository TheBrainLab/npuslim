"""Patch Ray worker rank adjustment to keep global rank in sync.

Root cause:
- ``RayDistributedExecutor`` creates workers in one order, then reorders them
  by node/IP and calls ``RayWorkerWrapper.adjust_rank()``.
- Upstream ``adjust_rank()`` updates ``rpc_rank`` only.
- ``WorkerWrapperBase.initialize_from_config()`` later indexes
  ``kv_cache_configs[self.global_rank]``.

With PP>1, different workers own different layer subsets, so a stale
``global_rank`` can make a worker receive another stage's KV-cache config and
fail during attention backend initialization.
"""

from __future__ import annotations

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import always_disable, package_version_range, register_patch


@register_patch(
    target="vllm.v1.executor.ray_utils",
    condition=package_version_range("vllm", max_version="0.20.1"),
    # condition=always_disable,
)
def patch_ray_worker_adjust_rank(module) -> None:
    """Keep ``global_rank`` aligned with reordered Ray worker ranks."""

    worker_cls = module.RayWorkerWrapper
    if getattr(worker_cls.adjust_rank, "_npuslim_patched", False):
        return

    original_adjust_rank = worker_cls.adjust_rank

    def patched_adjust_rank(self, rank_mapping: dict[int, int]) -> None:
        old_global_rank = self.global_rank
        original_adjust_rank(self, rank_mapping)

        if old_global_rank in rank_mapping:
            self.global_rank = rank_mapping[old_global_rank]

    patched_adjust_rank._npuslim_patched = True  # type: ignore[attr-defined]
    worker_cls.adjust_rank = patched_adjust_rank
    patch_logger.info(
        "Patched RayWorkerWrapper.adjust_rank to sync global_rank with rpc_rank"
    )
