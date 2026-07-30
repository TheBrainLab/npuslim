"""Patch registration: inject EnhancedNPUPrefetchOffloader into vllm-ascend.

This module is automatically discovered by npuslim's plugin system via
``discover_modules("npuslim.plugins.vllm_ascend", ...)`` and applied by
``apply_all_patches()``.

When ``npuslim_offload_trunk.enabled=True`` is set in vllm's additional_config
(or via ``NPUSLIM_OFFLOAD_TRUNK_ENABLED=1`` env var), this patch:

1. Computes a memory budget from model config and hardware
2. Plans which layers to offload (size-aware, group, or custom strategy)
3. Replaces the default offloader with EnhancedNPUPrefetchOffloader

If the offload trunk config is not enabled, the original behavior is
preserved — vllm-ascend's NPUPrefetchOffloader or vllm's default offloader
remains unchanged.
"""

from __future__ import annotations

from typing import Any

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import register_patch


@register_patch(target="vllm_ascend.worker.model_runner_v1")
def patch_model_runner_offloader(module: Any) -> None:
    """Patch NPUModelRunner to support npuslim offload trunk.

    Wraps ``__init__`` to detect npuslim offload config and prepare
    the planner. Wraps ``load_model`` to create and set the
    EnhancedNPUPrefetchOffloader before model loading begins.
    """

    NPUModelRunner = getattr(module, "NPUModelRunner", None)
    if NPUModelRunner is None:
        patch_logger.warning(
            "[OffloadTrunk] NPUModelRunner not found in "
            "vllm_ascend.worker.model_runner_v1, skipping patch"
        )
        return

    original_init = NPUModelRunner.__init__
    original_load_model = NPUModelRunner.load_model

    def patched_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        """Wrap __init__ to detect npuslim offload trunk config."""
        # Check if npuslim offload trunk is enabled BEFORE calling original_init.
        # We need to set vllm's offload_config before __init__ runs, because
        # gpu_model_runner.__init__ and NPUModelRunner.__init__ both check
        # offload_config to decide which offloader to create.
        from npuslim.plugins.vllm_ascend.offload.config import resolve_from_vllm_config

        offload_config = resolve_from_vllm_config(vllm_config)

        if not offload_config.enabled:
            # Not enabled — call original init with default config
            original_init(self, vllm_config, *args, **kwargs)
            self._npuslim_offload_config = None
            return

        if offload_config.backend != "prefetch":
            patch_logger.warning(
                f"[OffloadTrunk] backend='{offload_config.backend}' is not yet "
                f"supported, falling back to original offloader"
            )
            original_init(self, vllm_config, *args, **kwargs)
            self._npuslim_offload_config = None
            return

        # Store config for use in load_model
        self._npuslim_offload_config = offload_config

        # Pre-compute the offload plan so it's ready when make_layers() calls
        # get_offloader().wrap_modules()
        from npuslim.plugins.vllm_ascend.offload.memory_budget import (
            MemoryBudgetCalculator,
        )
        from npuslim.plugins.vllm_ascend.offload.planner import OffloadPlanner

        calculator = MemoryBudgetCalculator()
        planner = OffloadPlanner()

        budget = calculator.calculate(vllm_config, offload_config)
        plan = planner.plan(vllm_config, budget, offload_config)

        if not plan.offload_layer_indices:
            patch_logger.info(
                "[OffloadTrunk] No layers selected for offloading, "
                "keeping original offloader"
            )
            original_init(self, vllm_config, *args, **kwargs)
            self._npuslim_offload_config = None
            return

        # Set vllm's offload_config BEFORE original_init so the framework
        # creates the correct NPUPrefetchOffloader inside __init__.
        # This ensures the offloader is created at the right time in the
        # init sequence, matching the native --offload_backend prefetch path.
        num_offloaded = len(plan.offload_layer_indices)
        total_layers = plan.total_layers
        group_size = max(2, round(total_layers / num_offloaded))
        vllm_config.offload_config.offload_backend = "prefetch"
        vllm_config.offload_config.prefetch.offload_group_size = group_size
        vllm_config.offload_config.prefetch.offload_num_in_group = 1
        vllm_config.offload_config.prefetch.offload_prefetch_step = offload_config.prefetch_step
        vllm_config.offload_config.prefetch.offload_params = offload_config.offload_params

        patch_logger.info(
            f"[OffloadTrunk] Set vllm offload_config: group_size={group_size}, "
            f"num_in_group=1, prefetch_step={offload_config.prefetch_step}"
        )

        # Now call original init — it will create NPUPrefetchOffloader with
        # the group_size we just set, matching the native offload path.
        original_init(self, vllm_config, *args, **kwargs)

        # Replace the native NPUPrefetchOffloader with our Enhanced version.
        # The native offloader was created with uniform grouping; we replace
        # it with our plan-based selection. This is safe because load_model
        # (which calls wrap_modules) hasn't been called yet.

        # Create the enhanced offloader
        from npuslim.plugins.vllm_ascend.offload.monitor import OffloadMonitor
        from npuslim.plugins.vllm_ascend.offload.npu_prefetch_offloader import (
            EnhancedNPUPrefetchOffloader,
        )

        monitor = OffloadMonitor(
            log_interval=offload_config.monitor_log_interval
        ) if offload_config.enable_monitor else None

        enhanced_offloader = EnhancedNPUPrefetchOffloader(
            plan=plan,
            prefetch_step=offload_config.prefetch_step,
            offload_params=offload_config.offload_params,
            monitor=monitor,
        )

        # Replace the offloader BEFORE load_model is called.
        # make_layers() inside initialize_model() will call
        # get_offloader().wrap_modules(), so the enhanced offloader
        # must be set before model construction begins.
        from vllm.model_executor.offloader.base import set_offloader

        set_offloader(enhanced_offloader)

        self._npuslim_offload_monitor = monitor

        patch_logger.success(
            f"[OffloadTrunk] EnhancedNPUPrefetchOffloader installed: "
            f"{plan.summary()}"
        )

    def patched_load_model(self: Any) -> None:
        """Wrap load_model to handle offloaded layer weight processing.

        Problem: _CpuParamOffloader.__init__ (called during make_layers →
        wrap_modules) moves offloaded params to CPU. ascend_process_weights
        _after_loading has two phases:
          1. Process QuantizeMethodBase modules (with device_loading_context)
          2. Process Attention modules (with device_loading_context)
        Phase 1 moves fused_qkv_a_proj to NPU, processes it, moves it back to
        CPU. Phase 2 (MLAAttention) then accesses fused_qkv_a_proj.weight via
        self.impl — but fused_qkv_a_proj is NOT a registered submodule of
        MLAAttention (it's passed via **extra_impl_args to AscendSFAImpl
        which is not nn.Module), so device_loading_context can't move it back
        to NPU. npu_format_cast then fails on CPU tensor.

        Fix: replace process_weights_after_loading with a version that
        processes attention modules one layer at a time — move only the
        current layer's offloaded params to NPU, process, move back to CPU.
        This keeps peak HBM = resident weights + one layer (not all weights).

        Timing: wrap_modules (which populates offloader.module_offloaders)
        is called inside original_load_model → initialize_model, which runs
        BEFORE process_weights_after_loading. So we register the wrapper
        before calling original_load_model, and build layer_specs inside
        the wrapper (when it's actually called).
        """
        offload_config = getattr(self, "_npuslim_offload_config", None)
        if offload_config is None or not offload_config.enabled:
            original_load_model(self)
            return

        # For non-quantized models, call original_load_model then skip to
        # profiling (no need for wrapped_process_weights).
        model_config = getattr(self.vllm_config, "model_config", None)
        is_quantized = (
            getattr(model_config, "quantization", None) is not None
            or getattr(self.vllm_config, "quant_config", None) is not None
        )

        if not is_quantized:
            original_load_model(self)
        else:
            # Quantized models need per-layer process_weights_after_loading
            import vllm.model_executor.model_loader.base_loader as base_loader
            original_process_weights = base_loader.process_weights_after_loading

            def wrapped_process_weights(model, model_config, target_device):
                from vllm.model_executor.offloader.base import get_offloader
                from vllm.model_executor.layers.quantization.base_config import (
                    QuantizeMethodBase,
                )
                from vllm.model_executor.layers.attention import (
                    Attention, MLAAttention, MMEncoderAttention,
                )
                from vllm.model_executor.model_loader.utils import device_loading_context

                offloader = get_offloader()
                has_offload = (hasattr(offloader, "module_offloaders")
                               and offloader.module_offloaders)

                if not has_offload:
                    original_process_weights(model, model_config, target_device)
                    return

                # Build layer_idx → list of (param_offloader, device) mapping.
                # Available now because wrap_modules has already run.
                layer_specs: dict[int, list] = {}
                for mo in offloader.module_offloaders:
                    offloader_idx = mo.layer_idx
                    module_idx = offloader._offloader_idx_to_module_index.get(
                        offloader_idx, offloader_idx)
                    layer_specs[module_idx] = [
                        (po, mo.device) for po in mo._param_offloaders.values()
                    ]

                def _is_dsa_attention(mod):
                    cls = type(mod)
                    return (cls.__module__
                            == "vllm_ascend.models.layer.attention.layer"
                            and cls.__name__ == "DSAAttention")

                def _parse_layer_idx(module):
                    """Extract transformer layer index from an attention module."""
                    layer_name = getattr(module, "layer_name", None)
                    if layer_name:
                        parts = layer_name.split(".")
                        for i, p in enumerate(parts):
                            if p == "layers" and i + 1 < len(parts):
                                try:
                                    return int(parts[i + 1])
                                except ValueError:
                                    pass
                    return None

                # Phase 1: QuantizeMethodBase modules — device_loading_context
                # handles per-module NPU↔CPU correctly (moves one module's
                # params at a time, not all at once).
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if isinstance(quant_method, QuantizeMethodBase):
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)

                # Phase 2: Attention modules — process one layer at a time.
                # For offloaded layers, move ONLY that layer's params to NPU
                # before processing, then back to CPU after. This keeps
                # peak HBM = resident + one offloaded layer.
                for _, module in model.named_modules():
                    is_attn = (isinstance(module, (Attention, MLAAttention,
                                                    MMEncoderAttention))
                               or _is_dsa_attention(module))
                    if not is_attn or not hasattr(
                            module, "process_weights_after_loading"):
                        continue

                    layer_idx = _parse_layer_idx(module)
                    specs = layer_specs.get(layer_idx) if layer_idx is not None else None

                    if specs:
                        for po, device in specs:
                            param = po._param
                            if param.data.device.type == "cpu":
                                param.data = param.data.to(device)

                    with device_loading_context(module, target_device):
                        module.process_weights_after_loading(model_config.dtype)

                    if specs:
                        for po, device in specs:
                            param = po._param
                            if param.data.device.type != "cpu":
                                param.data = param.data.to("cpu")

                if model_config.quantization == "torchao":
                    from vllm.model_executor.model_loader.reload import (
                        set_torchao_reload_attrs,
                    )
                    set_torchao_reload_attrs(model, model_config)

            base_loader.process_weights_after_loading = wrapped_process_weights

            try:
                original_load_model(self)
            finally:
                base_loader.process_weights_after_loading = original_process_weights

        # Log final monitor report
        monitor = getattr(self, "_npuslim_offload_monitor", None)
        if monitor is not None:
            patch_logger.info(f"[OffloadTrunk] {monitor.final_report()}")

    NPUModelRunner.__init__ = patched_init
    NPUModelRunner.load_model = patched_load_model

    patch_logger.success(
        "[OffloadTrunk] Patched NPUModelRunner.__init__ and load_model "
        "for npuslim offload trunk support"
    )
