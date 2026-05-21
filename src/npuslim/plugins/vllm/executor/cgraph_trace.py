"""Trace CGRAPH worker entry/exit to diagnose EP hang.

Currently disabled — the EP hang was traced to Ray CGRAPH / vllm-Ascend
collective-operation interaction, not to the MoE plugin layer.
"""

from __future__ import annotations

from npuslim.plugins.registry import always_disable, register_patch


@register_patch(target="vllm.v1.executor.ray_utils", condition=always_disable)
def patch_cgraph_trace(module) -> None:
    pass


@register_patch(target="vllm.worker.worker", condition=always_disable)
def patch_model_runner_trace(module) -> None:
    pass
