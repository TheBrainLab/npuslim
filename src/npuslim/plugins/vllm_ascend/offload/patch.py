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

        # Quantized weights (W4A8 modelslim) are repacked to FRACTAL_NZ in
        # process_weights_after_loading via maybe_trans_nz. torch-npu
        # silently demotes internal-format tensor creation to ND unless the
        # ALLOW_INTERNAL_FORMAT option is enabled — ND weights are then
        # rejected by aclnnGroupedMatmul*WeightNz ops ("storageShape must be
        # 5, got [N], dimNum is 1"). Enable it here — before original_init,
        # i.e. before any weight loading/processing — whenever the offload
        # trunk is active. (vllm_config.quant_config is populated lazily
        # during config verification, so it cannot be used as a gate here.)
        try:
            import torch_npu

            if torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") != b"enable":
                torch_npu.config.allow_internal_format = True
                patch_logger.info(
                    "[OffloadTrunk] ALLOW_INTERNAL_FORMAT enabled "
                    "(FRACTAL_NZ weights)"
                )
        except Exception as e:
            patch_logger.warning(
                f"[OffloadTrunk] failed to enable ALLOW_INTERNAL_FORMAT: {e}"
            )

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

        # Dedicated NZ slots are only needed by the graph-capture path.
        # Auto-disable when cudagraph_mode=NONE: the eager path refills
        # every step anyway, and per-module slots would duplicate ALL
        # offloaded NZ weights in HBM (net savings -> 0, OOM for all-NZ
        # models like K2.6 — incident 2026-08-21). Config can override.
        dedicated_slots = offload_config.dedicated_nz_slots
        if dedicated_slots is None:
            try:
                cg_mode = vllm_config.compilation_config.cudagraph_mode
                dedicated_slots = "NONE" not in str(cg_mode)
            except Exception:
                dedicated_slots = True
            patch_logger.info(
                f"[OffloadTrunk] cudagraph_mode={cg_mode} -> "
                f"dedicated_nz_slots={dedicated_slots}"
            )

        enhanced_offloader = EnhancedNPUPrefetchOffloader(
            plan=plan,
            prefetch_step=offload_config.prefetch_step,
            offload_params=offload_config.offload_params,
            monitor=monitor,
            dedicated_slots=dedicated_slots,
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
        wrap_modules) moves offloaded params to CPU. process_weights_after
        _loading has two phases:
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
        before calling original_load_model, and collect the offloaded param
        offloaders inside the wrapper (when it's actually called).

        Note on interaction with vllm-ascend: vllm-ascend already patches
        base_loader.process_weights_after_loading with
        ascend_process_weights_after_loading (a superset of the upstream
        version that adds DSAAttention support). Our wrapped version keeps
        all of that, adds update_param_tp_status after quant-method
        processing, and adds the per-layer NPU↔CPU transfer needed for
        offload scenarios.

        CAVEAT: We access private attributes (module_offloaders,
        _offloader_idx_to_module_index, _param_offloaders, _param) on the
        offloader and its sub-objects. These are not part of the public API
        and may break across vllm/vllm-ascend version upgrades.
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
                import torch
                from vllm.model_executor.offloader.base import get_offloader
                from vllm.model_executor.layers.quantization.base_config import (
                    QuantizeMethodBase,
                )
                from vllm.model_executor.layers.attention import (
                    Attention, MLAAttention, MMEncoderAttention,
                )
                from vllm.model_executor.model_loader.utils import (
                    device_loading_context,
                )

                offloader = get_offloader()
                module_offloaders = getattr(offloader, "module_offloaders", None)
                has_offload = module_offloaders and len(module_offloaders) > 0

                if not has_offload:
                    original_process_weights(model, model_config, target_device)
                    return

                # Flatten all offloaded param offloaders across layers.
                # Available now because wrap_modules has already run.
                all_param_offloaders: list = []
                for mo in module_offloaders:
                    param_offloaders = getattr(mo, "_param_offloaders",
                                        getattr(mo, "param_offloaders", {}))
                    all_param_offloaders.extend(param_offloaders.values())

                def _param_of(po):
                    """Current nn.Parameter of a param offloader, or None if
                    process_weights_after_loading deleted it (transient scale
                    params are dropped after being absorbed into permanent
                    buffers)."""
                    try:
                        return po._param
                    except AttributeError:
                        return None

                def _is_dsa_attention(mod):
                    """Check if module is DSAAttention.

                    Prefer isinstance when the class is importable; fall back
                    to class name matching to handle environments where the
                    vllm-ascend attention layer module is not installed.
                    """
                    try:
                        from vllm_ascend.models.layer.attention.layer import (
                            DSAAttention,
                        )
                        return isinstance(mod, DSAAttention)
                    except ImportError:
                        cls = type(mod)
                        return (cls.__module__
                                == "vllm_ascend.models.layer.attention.layer"
                                and cls.__name__ == "DSAAttention")

                def _d2h_probe(tag: str, tensor: Any = None) -> None:
                    """Probe whether NPU→CPU copies work right now."""
                    t = tensor if tensor is not None else torch.ones(
                        4, device=target_device)
                    try:
                        t.to("cpu")
                        torch.npu.synchronize(target_device)
                        patch_logger.info(f"[OffloadTrunk] D2H probe ({tag}): OK")
                    except Exception as e:
                        patch_logger.warning(
                            f"[OffloadTrunk] D2H probe ({tag}) FAILED: "
                            f"{type(e).__name__}: {str(e)[:160]}")

                _d2h_probe("entry")

                # Offloaded params are moved to the NPU BEFORE weight
                # processing, so device_loading_context finds nothing on
                # CPU and performs no NPU→CPU copy during the phases below.
                # In DCP/EP worker contexts, D2H copies fail with
                # "aclrtAllocatorGetByStream failed: The stream is not
                # registered with any allocator" once quantization repack
                # ops (npu_format_cast) have run on the default stream, so
                # the processing phases run on a dedicated side stream and
                # the final bulk D2H runs on the untouched default stream.
                for po in all_param_offloaders:
                    param = _param_of(po)
                    if param is not None and param.data.device.type == "cpu":
                        param.data = param.data.to(target_device)

                proc_stream = torch.npu.Stream()
                with torch.npu.stream(proc_stream):
                    # Phase 1: QuantizeMethodBase modules — offloaded params
                    # are already on the NPU, so device_loading_context
                    # performs no copies for them (resident modules are
                    # untouched as well).
                    #
                    # Note: vllm-ascend's _ModuleOffloader has already
                    # wrapped quant_method.process_weights_after_loading with
                    # NZ format detection
                    # (_capture_static_buffer_formats_from_npu_params), so
                    # calling it here also triggers that detection correctly.
                    for _, module in model.named_modules():
                        quant_method = getattr(module, "quant_method", None)
                        if isinstance(quant_method, QuantizeMethodBase):
                            with device_loading_context(module, target_device):
                                quant_method.process_weights_after_loading(module)
                            # Reconcile TP state after quant_method may have
                            # swapped in fresh Parameters (e.g. FP8 requant).
                            if hasattr(module, "update_param_tp_status"):
                                module.update_param_tp_status()

                    # Phase 2: Attention modules (incl. DSA) — same no-copy
                    # situation as phase 1.
                    for _, module in model.named_modules():
                        is_attn = (isinstance(module, (Attention, MLAAttention,
                                                        MMEncoderAttention))
                                   or _is_dsa_attention(module))
                        if not is_attn or not hasattr(
                                module, "process_weights_after_loading"):
                            continue
                        with device_loading_context(module, target_device):
                            module.process_weights_after_loading(model_config.dtype)

                # Order the default stream after the processing stream, then
                # move offloaded params back to CPU (release HBM) so post_init
                # (sync_cpu_storage / static buffer assignment) observes the
                # final CPU tensors.
                #
                # In DCP/EP worker contexts large D2H copies fail with
                # "aclrtAllocatorGetByStream failed: The stream is not
                # registered with any allocator" (the TransData runtime
                # staging allocation cannot resolve the stream's allocator),
                # while small copies succeed. Each param is therefore copied
                # directly first and falls back to 32MB chunked copies.
                def _d2h_chunked(src: torch.Tensor) -> torch.Tensor:
                    """D2H via flat 32MB slices (small TransData copies that
                    do not need runtime staging buffers).

                    The destination must be a freshly allocated N-D tensor:
                    AIV ops (e.g. aclnnGroupedMatmulSwigluQuantWeightNzV2)
                    reject scale tensors whose storageShape is 1-D, which is
                    what a flat-buffer .view(src.shape) reports to the ACL
                    plugin. Allocating torch.empty(src.shape, device="cpu")
                    matches the N-D storage semantics of a normal .to("cpu").
                    """
                    dst = torch.empty(src.shape, dtype=src.dtype, device="cpu")
                    flat_src = src.flatten()
                    flat_dst = dst.flatten()
                    step = max(1, (32 * 1024 * 1024) // flat_src.element_size())
                    for i in range(0, flat_src.numel(), step):
                        flat_dst[i:i + step].copy_(flat_src[i:i + step])
                    return dst

                torch.npu.current_stream(target_device).wait_stream(proc_stream)
                _d2h_probe("pre-d2h-big-int8", torch.ones(
                    512 * 1024 * 1024, dtype=torch.int8, device=target_device))
                d2h_failed: list = []
                for idx, po in enumerate(all_param_offloaders):
                    param = _param_of(po)
                    if param is None or param.data.device.type == "cpu":
                        continue
                    d = param.data
                    pname = getattr(po, "_param_name", "") or ""
                    # FRACTAL_NZ weights (W4A8 w13/w2 after maybe_trans_nz)
                    # cannot round-trip through CPU as-is:
                    #  - .to("cpu") on the int32 NZ view crashes (Identity
                    #    TransData tiling failure)
                    #  - the aclnnGroupedMatmul*WeightNz ops reject buffers
                    #    whose storageShape is not the 5-D NZ physical layout
                    # Hardware-verified pipeline (all format casts on the
                    # int8 base — casting the int32 view crashes):
                    #   view(i8) -> npu_format_cast(ND) -> .to("cpu")
                    # The name-based fallback covers packed w13/w2 even when
                    # get_npu_format does not report the internal format.
                    is_nz = False
                    pre_fmt = None
                    try:
                        import torch_npu

                        pre_fmt = torch_npu.get_npu_format(d)
                        is_nz = "NZ" in str(pre_fmt)
                        if (not is_nz and d.dtype == torch.int32
                                and pname.endswith(
                                    ("w13_weight", "w2_weight"))):
                            is_nz = True
                    except Exception:
                        if (d.dtype == torch.int32
                                and pname.endswith(
                                    ("w13_weight", "w2_weight"))):
                            is_nz = True
                    size_mb = d.numel() * d.element_size() / 1024**2
                    if size_mb > 4.0:
                        patch_logger.info(
                            f"[OffloadTrunk] preD2H param[{idx}] {pname} "
                            f"{d.dtype} {tuple(d.shape)} fmt={pre_fmt} "
                            f"nz={is_nz}")
                    if is_nz:
                        if pname:
                            offloader.nz_param_keys.add(pname)

                        def _nz_d2h(chunked: bool = False) -> torch.Tensor:
                            import torch_npu

                            i8 = (d.view(torch.int8)
                                  if d.dtype == torch.int32 else d)
                            nd = torch_npu.npu_format_cast(i8, 2)  # ND
                            if chunked:
                                cpu = _d2h_chunked(nd)
                            else:
                                cpu = nd.to("cpu")
                                if tuple(cpu.shape) != tuple(nd.shape):
                                    cpu = cpu.reshape(nd.shape)
                            return (cpu.view(torch.int32)
                                    if d.dtype == torch.int32 else cpu)

                        try:
                            param.data = _nz_d2h()
                            patch_logger.info(
                                f"[OffloadTrunk] D2H param[{idx}] OK nz: "
                                f"{size_mb:.1f}MB {d.dtype} "
                                f"{tuple(d.shape)} -> cpu "
                                f"{tuple(param.data.shape)}")
                        except Exception as e:
                            patch_logger.warning(
                                f"[OffloadTrunk] D2H param[{idx}] nz direct "
                                f"FAILED: {size_mb:.1f}MB: "
                                f"{str(e)[:100]} — retrying nz chunked")
                            try:
                                param.data = _nz_d2h(chunked=True)
                                patch_logger.info(
                                    f"[OffloadTrunk] D2H param[{idx}] OK "
                                    f"nz-chunked: {size_mb:.1f}MB {d.dtype}")
                            except Exception as e2:
                                patch_logger.error(
                                    f"[OffloadTrunk] D2H param[{idx}] "
                                    f"nz-chunked FAILED: {size_mb:.1f}MB: "
                                    f"{str(e2)[:100]}")
                                d2h_failed.append(idx)
                        continue
                    try:
                        param.data = d.to("cpu")
                        patch_logger.info(
                            f"[OffloadTrunk] D2H param[{idx}] OK direct: "
                            f"{size_mb:.1f}MB {d.dtype} {tuple(d.shape)} -> cpu "
                            f"{tuple(param.data.shape)} stride={param.data.stride()}")
                    except Exception as e:
                        patch_logger.warning(
                            f"[OffloadTrunk] D2H param[{idx}] direct FAILED: "
                            f"{size_mb:.1f}MB {d.dtype} {tuple(d.shape)}: "
                            f"{str(e)[:100]} — retrying chunked")
                        try:
                            param.data = _d2h_chunked(d)
                            patch_logger.info(
                                f"[OffloadTrunk] D2H param[{idx}] OK chunked: "
                                f"{size_mb:.1f}MB {d.dtype}")
                        except Exception as e2:
                            patch_logger.error(
                                f"[OffloadTrunk] D2H param[{idx}] chunked "
                                f"FAILED: {size_mb:.1f}MB {d.dtype}: "
                                f"{str(e2)[:100]}")
                            d2h_failed.append(idx)
                if d2h_failed:
                    raise RuntimeError(
                        f"[OffloadTrunk] D2H failed for {len(d2h_failed)} "
                        f"params: {d2h_failed[:10]}")
                torch.npu.synchronize(target_device)

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
