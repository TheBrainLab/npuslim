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
from vllm.model_executor.offloader.prefetch import StaticBufferPool
from vllm_ascend.model_executor.offloader.prefetch import NPUPrefetchOffloader

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.vllm_ascend.offload.monitor import OffloadMonitor
from npuslim.plugins.vllm_ascend.offload.planner import OffloadPlan

# ACL_FORMAT_FRACTAL_NZ (vllm_ascend.utils constant)
_ACL_FORMAT_FRACTAL_NZ = 29


def _trace_enabled() -> bool:
    return os.environ.get("NPUSLIM_OFFLOAD_TRACE", "0") in ("1", "true", "True")


class _NZStaticBufferPool(StaticBufferPool):
    """StaticBufferPool that materializes FRACTAL_NZ slots for W4A8 weights.

    aclnnGroupedMatmul*WeightNz ops require the weight input to be a genuine
    internal-format tensor; a plain ND slot (empty_strided) is rejected with
    "storageShape must be 5, got [N], dimNum is 1". For keys whose param name
    is in ``nz_names`` the slot is rebuilt as an NZ tensor: for int32 weights
    (pack_to_int32 output) the NZ cast is done on the equivalent int8 buffer
    and viewed back to int32 (the view preserves the internal format, as does
    the load-time pack_to_int32 path). Hardware-verified: slot.copy_(nd_cpu)
    is a supported ND->NZ cross-format copy that keeps the slot FRACTAL_NZ.
    """

    nz_names: frozenset = frozenset()

    # Number of dedicated slots per NZ key (= number of offloaded modules).
    # Each offloaded layer gets its OWN NZ slot so no data movement is
    # needed inside NPU graph capture (any internal-format op-plugin op —
    # view, cross-format copy_, npu_format_cast — dispatches an "Identity"
    # aclop that acl_graph capture rejects).
    nz_slot_count: int = 1

    # data_ptr(int32-slot view) -> int8 FRACTAL_NZ base tensor. Filled
    # eagerly in post_init; consulted at onload time so no NPU-side view()
    # runs there.
    i8_bases: dict = {}

    def __init__(
        self,
        param_infos: list,
        slot_capacity: int,
        device: torch.device,
    ):
        super().__init__(param_infos, slot_capacity, device)
        if not self.nz_names:
            self._nz_assign_count = {}
            return
        self._nz_assign_count = {}
        for key, slots in list(self._buffers.items()):  # noqa: SLF001
            if key[0] not in self.nz_names:
                continue
            shape, dtype = key[1], key[3]
            for i in range(len(slots)):
                slots[i] = self._make_nz_slot(shape, dtype, device)
            # Extend with the dedicated slots (one per offloaded module).
            while len(slots) < self.nz_slot_count:
                slots.append(self._make_nz_slot(shape, dtype, device))

    def get_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        # NZ keys: assign a UNIQUE slot per requesting module (modules are
        # assigned in order, each requesting every key exactly once), so
        # slots are never recycled across layers — the captured graph then
        # needs no NZ refill. Non-NZ keys keep the native circular scheme.
        if name in self.nz_names:
            key = (name, shape, stride, dtype)
            i = self._nz_assign_count.get(key, 0)
            self._nz_assign_count[key] = i + 1
            return self._buffers[key][i % len(self._buffers[key])]  # noqa: SLF001
        return super().get_buffer(name, shape, stride, dtype, slot_idx)

    @classmethod
    def _make_nz_slot(cls, shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
        import torch_npu

        if dtype == torch.int32 and shape and shape[-1] % 4 == 0:
            nz = torch_npu.npu_format_cast(
                torch.zeros(
                    *shape[:-1],
                    shape[-1] * 4,
                    dtype=torch.int8,
                    device=device,
                ),
                _ACL_FORMAT_FRACTAL_NZ,
            )
            viewed = nz.view(torch.int32)
            cls.i8_bases[viewed.data_ptr()] = nz
            return viewed
        nz = torch_npu.npu_format_cast(
            torch.zeros(shape, dtype=dtype, device=device),
            _ACL_FORMAT_FRACTAL_NZ,
        )
        cls.i8_bases[nz.data_ptr()] = nz
        return nz


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

        # Param names whose NPU-side tensors carry the FRACTAL_NZ internal
        # format (W4A8 w13/w2 after maybe_trans_nz). Populated by the
        # process_weights wrapper during D2H (the last moment the NPU format
        # is observable); consumed by post_init to build NZ pool slots.
        self.nz_param_keys: set[str] = set()

        # data_ptr(slot) -> int8 FRACTAL_NZ base, filled in post_init.
        # Avoids view() on internal-format tensors at onload time (view
        # dispatches an Identity aclop, illegal during NPU graph capture).
        self._nz_i8_handles: dict[int, torch.Tensor] = {}

        # Map from original module_index -> offloader index
        self._module_index_to_offloader_idx: dict[int, int] = {}
        self._offloader_idx_to_module_index: dict[int, int] = {}
        self._module_index_to_name: dict[int, str] = {}

    def post_init(self):
        """Parent post_init with FRACTAL_NZ buffer slots for quantized weights.

        aclnnGroupedMatmul*WeightNz ops reject weights whose storage is not a
        genuine internal-format tensor ("storageShape must be 5"). The parent
        creates plain ND slots via empty_strided, so for keys in
        ``nz_param_keys`` we swap in _NZStaticBufferPool while super().post_init()
        runs (the pool class is looked up in the vllm_ascend prefetch module
        namespace at call time).
        """
        if not self.nz_param_keys:
            super().post_init()
            self._log_buffer_diagnostics()
            return

        import vllm_ascend.model_executor.offloader.prefetch as va_prefetch

        pool_cls = type(
            "_NZStaticBufferPoolForParams", (_NZStaticBufferPool,), {}
        )
        pool_cls.nz_names = frozenset(self.nz_param_keys)
        # Dedicated slot per offloaded module (see _NZStaticBufferPool).
        pool_cls.nz_slot_count = len(self.module_offloaders)
        orig_pool_cls = va_prefetch.StaticBufferPool
        _NZStaticBufferPool.i8_bases.clear()
        try:
            va_prefetch.StaticBufferPool = pool_cls
            super().post_init()
        finally:
            va_prefetch.StaticBufferPool = orig_pool_cls
        # Snapshot the eagerly-built int8 base handles for capture-safe onload.
        self._nz_i8_handles = dict(_NZStaticBufferPool.i8_bases)

        nz_slots = sum(
            1
            for key in self.buffer_pool._buffers  # noqa: SLF001
            if key[0] in self.nz_param_keys
        )
        patch_logger.info(
            f"[EnhancedNPUPrefetchOffloader] FRACTAL_NZ pool slots for "
            f"{nz_slots} unique param keys: {sorted(self.nz_param_keys)}"
        )
        self._log_buffer_diagnostics()

    def _log_buffer_diagnostics(self) -> None:
        """Log buffer / cpu_storage layouts for large offloaded params."""
        pool = getattr(self, "buffer_pool", None)
        patch_logger.info(
            f"[EnhancedNPUPrefetchOffloader] post_init nz_param_keys="
            f"{len(self.nz_param_keys)} {sorted(self.nz_param_keys)[:10]}")
        if pool is None:
            return
        import torch_npu

        try:
            from torchair.core import _npu_graph_executor as _ex
        except Exception:
            _ex = None
        for key, slots in pool._buffers.items():  # noqa: SLF001
            buf = slots[0]
            if buf.numel() <= 1_000_000:
                continue
            storage = _ex.GetNpuStorageSizes(buf) if _ex is not None else "?"
            try:
                fmt = torch_npu.get_npu_format(buf)
            except Exception:
                fmt = "?"
            patch_logger.info(
                f"[EnhancedNPUPrefetchOffloader] bufDiag {key[0]} "
                f"keyShape={key[1]} bufShape={tuple(buf.shape)} "
                f"storage={storage} fmt={fmt}")
        for mo in self.module_offloaders:
            for pname, po in mo._param_offloaders.items():
                cs = getattr(po, "_cpu_storage", None)
                if cs is not None and cs.numel() > 1_000_000:
                    patch_logger.info(
                        f"[EnhancedNPUPrefetchOffloader] cpuDiag {pname} "
                        f"shape={tuple(cs.shape)} pinned={cs.is_pinned()}")

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

            mo = _NPUModuleOffloader(
                mode=self.mode,
                module=module,
                copy_stream=self.copy_stream,
                whitelist_param_names=whitelist,
                layer_idx=offloader_idx,
            )
            self._patch_module_onload(mo)
            self.module_offloaders.append(mo)

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

    def _nz_onload(self, buf: torch.Tensor, cpu: torch.Tensor) -> None:
        """Refill a FRACTAL_NZ static buffer from ND CPU storage (EAGER only).

        ``buf`` is either a FRACTAL_NZ int8 tensor or the int32 view of an
        int8 FRACTAL_NZ base (as built by _NZStaticBufferPool); ``cpu``
        holds the matching ND-ordered bytes. A single int8-level
        cross-format copy_ converts the layout correctly (hardware-verified,
        temp/test_nz_rawbytes.py: int32-level copies corrupt, int8-level
        does not). Never called during graph capture — see
        _patch_module_onload.
        """
        # Live i8_bases dict: populated when the pool is created, i.e.
        # BEFORE the first onload. Lazily derive + cache fallback (eager).
        buf_i8 = _NZStaticBufferPool.i8_bases.get(buf.data_ptr())
        if buf_i8 is None:
            buf_i8 = buf if buf.dtype == torch.int8 else buf.view(torch.int8)
            _NZStaticBufferPool.i8_bases[buf.data_ptr()] = buf_i8
        cpu_i8 = cpu if cpu.dtype == torch.int8 else cpu.view(torch.int8)
        buf_i8.copy_(cpu_i8, non_blocking=True)

    def _patch_module_onload(self, mo) -> None:
        """Replace the module offloader's onload with an NZ-aware version.

        - EAGER: NZ params refill via one int8-level cross-format copy_
          (correct layout conversion; the int32-level ND->NZ copy corrupts
          data, hardware-verified).
        - CAPTURE: NZ params need NO data movement. Every offloaded module
          owns a DEDICATED NZ slot (see _NZStaticBufferPool) filled during
          the eager phase (post_init prefetch + profile_run), so the
          captured graph only records the no-op event protocol. Any
          internal-format op-plugin op (view, cross-format copy_,
          npu_format_cast) dispatches an "Identity" aclop, which the
          acl_graph capture path rejects (rounds 11b/15/16-retry).
          Non-NZ params keep the native capturable ND copy_ (circular
          slots, refilled on every replay as in the native design).
        """
        orig = mo.start_onload_to_static
        mo._nz_slots_loaded = False

        def patched():
            if not any(name in self.nz_param_keys
                       for name in mo._param_offloaders):
                orig()
                return
            capturing = torch.npu.is_current_stream_capturing()
            if capturing and mo._nz_slots_loaded:
                # Dedicated NZ slots already filled: capture the (no-op)
                # event protocol only, no data movement.
                mo._prefetch_in_capture = True
                mo._copy_done_event.record(mo.copy_stream)
                mo._event_valid_for_eager = False
                return
            mo._prefetch_in_capture = capturing
            fork_event = torch.npu.Event()
            torch.npu.current_stream().record_event(fork_event)
            mo.copy_stream.wait_event(fork_event)
            with torch.npu.stream(mo.copy_stream):
                for name, po in mo._param_offloaders.items():
                    cpu = po._cpu_storage
                    buf = po._gpu_buffer
                    if cpu is None or buf is None:
                        continue
                    if name in self.nz_param_keys:
                        self._nz_onload(buf, cpu)
                    else:
                        buf.copy_(cpu, non_blocking=True)
            mo._copy_done_event.record(mo.copy_stream)
            mo._event_valid_for_eager = not capturing
            mo._nz_slots_loaded = True

        mo.start_onload_to_static = patched
