"""Enhanced NPU PrefetchOffloader with intelligent layer selection.

Extends vllm-ascend's NPUPrefetchOffloader to support:
- Custom offload layer selection (not limited to uniform grouping)
- Integration with OffloadPlan for size-aware strategy
- Runtime monitoring via OffloadMonitor
- Detailed prefetch tracing (controlled by NPUSLIM_OFFLOAD_TRACE env var)
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any, Optional, Set

import torch
import torch.nn as nn

# Import the upstream NPU offloader to inherit all its machinery
from vllm_ascend.model_executor.offloader.prefetch import NPUPrefetchOffloader

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.vllm_ascend.offload.monitor import OffloadMonitor
from npuslim.plugins.vllm_ascend.offload.planner import OffloadPlan


def _trace_enabled() -> bool:
    return os.environ.get("NPUSLIM_OFFLOAD_TRACE", "0") in ("1", "true", "True")


class EnhancedNPUPrefetchOffloader(NPUPrefetchOffloader):
    """NPU PrefetchOffloader with custom layer selection via OffloadPlan.

    Inherits ALL stream/event/capture logic from NPUPrefetchOffloader.
    Only overrides __init__ and wrap_modules for plan-based layer selection.
    """

    def __init__(
        self,
        plan: OffloadPlan,
        prefetch_step: Optional[int] = None,
        offload_params: Optional[Set[str]] = None,
        mode: str = "cpu",
        monitor: Optional[OffloadMonitor] = None,
    ):
        # Pass dummy group_size/num_in_group to parent — we override
        # wrap_modules so they are never used.
        super().__init__(
            group_size=1,
            num_in_group=1,
            prefetch_step=prefetch_step or plan.prefetch_step,
            offload_params=offload_params or set(),
            mode=mode,
        )

        self.plan = plan
        self.monitor = monitor

        # Map from original module_index -> offloader index
        self._module_index_to_offloader_idx: dict[int, int] = {}
        self._offloader_idx_to_module_index: dict[int, int] = {}
        self._module_index_to_name: dict[int, str] = {}

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with prefetch offloading based on OffloadPlan.

        Only layers whose index appears in ``plan.offload_layer_indices``
        are offloaded. All other layers pass through unchanged.
        """
        from vllm_ascend.model_executor.offloader.prefetch import (
            _NPUModuleOffloader,
        )

        assert len(self.module_offloaders) == 0

        all_modules: list[nn.Module] = []

        for module_index, module in enumerate(modules_generator):
            all_modules.append(module)
            self._module_index_to_name[module_index] = f"model.layers.{module_index}"

            if module_index not in self.plan.offload_layer_indices:
                continue

            if self.offload_params:
                whitelist = [
                    name
                    for name, _ in module.named_parameters()
                    if any(f".{p}." in f".{name}." for p in self.offload_params)
                ]
            else:
                whitelist = [name for name, _ in module.named_parameters()]

            if not whitelist:
                continue

            offloader_idx = len(self.module_offloaders)
            self._module_index_to_offloader_idx[module_index] = offloader_idx
            self._offloader_idx_to_module_index[offloader_idx] = module_index

            self.module_offloaders.append(
                _NPUModuleOffloader(
                    mode=self.mode,
                    module=module,
                    copy_stream=self.copy_stream,
                    whitelist_param_names=whitelist,
                    layer_idx=offloader_idx,
                )
            )

        # Hook each offloaded module's forward (inherited from parent)
        for index, module in enumerate(
            all_modules[i] for i in sorted(self._module_index_to_offloader_idx.keys())
        ):
            self._hook_module_forward(index, module)

        if self.monitor:
            self.monitor.record_plan(
                total_layers=len(all_modules),
                offloaded_layers=len(self.module_offloaders),
                plan=self.plan,
            )

        patch_logger.info(
            f"[EnhancedNPUPrefetchOffloader] "
            f"Offloaded {len(self.module_offloaders)}/{len(all_modules)} layers, "
            f"prefetch_step={self.prefetch_step}"
        )

        return all_modules
