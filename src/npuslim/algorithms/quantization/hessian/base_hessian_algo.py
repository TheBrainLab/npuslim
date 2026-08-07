"""Shared runtime infrastructure for Hessian-based quantization algorithms."""

from __future__ import annotations

import gc
import os
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_module_tensor_to_device
from loguru import logger
from tqdm import tqdm

from npuslim.algorithms.quantization.base_quant_algo import BaseQuantizationAlgorithm
from npuslim.algorithms.quantization.hessian.hessian_common import (
    BaseHessianModule,
    _get_child_module,
    _unwrap_output,
)
from npuslim.core.backend import bh


class BaseHessianAlgorithm(BaseQuantizationAlgorithm):
    """Common chunk/runtime workflow for Hessian-based algorithms."""

    _TAG = "Hessian"
    _quantized_type_label = "Hessian"

    def __init__(self, max_calib_samples: int = 128, quantize_mtp: bool = False, save_mtp_debug: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.max_calib_samples = max(int(max_calib_samples), 1)
        self._calib_batch_size = 1

        self._runtime_model: Optional[nn.Module] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_state_keys: set[str] = set()
        self._block_name: str = "model.layers"
        self._total_layers: int = 0
        self._next_expected_layer_index: Optional[int] = None
        self._inps: Optional[torch.Tensor] = None
        self._outs: Optional[torch.Tensor] = None
        self._layer_kwargs: Dict[str, Any] = {}
        # DSA (Dynamic Sparse Attention) support: track prev_topk_indices
        # across layers. GLM-5's "shared" indexer layers require topk_indices
        # from a previous "full" indexer layer. In streaming mode (layer-by-layer),
        # we must propagate this manually.
        self._prev_topk_indices: Optional[torch.Tensor] = None

        # MTP (Multi-Token Prediction) support
        self._quantize_mtp = bool(quantize_mtp)
        self._save_mtp_debug = bool(save_mtp_debug)
        self._mtp_layer_names: List[str] = []
        self._mtp_embeds: Optional[torch.Tensor] = None

    @property
    def _ascend_quant_type(self) -> str:
        return ""

    def _log_start_params(self) -> None:
        pass

    def _create_handlers(self, layer_module: nn.Module, targets) -> Dict[str, BaseHessianModule]:
        raise NotImplementedError

    def _process_layer_handlers(self, layer, targets, handlers, chunk) -> tuple[set[str], int]:
        raise NotImplementedError

    def _update_quantization_metadata(self) -> None:
        pass

    def _finalize_chunk_metadata(self, chunk, quantized_tensor_names: set[str]) -> None:
        tensor_types = {name: "FLOAT" for name in chunk.all_tensors().keys()}
        quantized_type = self._ascend_quant_type if self.target_backend == "npu" else self._quantized_type_label
        for name in quantized_tensor_names:
            if name in tensor_types:
                tensor_types[name] = quantized_type
        chunk.metadata["tensor_types"] = tensor_types

    def on_start(self) -> None:
        self._log_start_params()
        if self._model_obj is None:
            raise ValueError(f"[{self._TAG}] runtime model is required")

        self._runtime_model = self._build_runtime_model()
        runtime_cfg = getattr(self._runtime_model, "config", None)
        if runtime_cfg is not None and hasattr(runtime_cfg, "use_cache"):
            try:
                runtime_cfg.use_cache = False
            except Exception:
                pass
        self._runtime_state_keys = set(self._runtime_model.state_dict().keys())
        self._block_name = getattr(
            self._model_obj,
            "block_name",
            getattr(self._model_obj, "layers_path", "model.layers"),
        )
        layer_container = _get_child_module(self._runtime_model, self._block_name)
        self._total_layers = len(layer_container)
        self._runtime_device = None
        self._next_expected_layer_index = None
        self._inps = None
        self._outs = None
        self._layer_kwargs = {}
        self._calib_batch_size = 1
        self._prev_topk_indices = None
        self._fallback_events: list[tuple[str, str]] = []  # (layer_name, fallback_type)

        # Detect MTP layers from model wrapper
        self._mtp_layer_names = list(getattr(self._model_obj, "mtp_layer_names", []))
        if self._quantize_mtp and self._mtp_layer_names:
            logger.info(
                f"[{self._TAG}] MTP quantization enabled for layers: {self._mtp_layer_names}"
            )
        elif self._quantize_mtp:
            logger.warning(f"[{self._TAG}] quantize_mtp=True but no MTP layers found")

    def on_finish(self) -> None:
        self._update_quantization_metadata()
        # Log Hessian fallback summary before cleaning up
        if self._fallback_events:
            layer_summary = ", ".join(
                f"{lname}({fbtype})" for lname, fbtype in self._fallback_events
            )
            logger.warning(
                f"[{self._TAG}] Hessian fallback summary: {len(self._fallback_events)} module(s) "
                f"used fallback strategies during quantization. "
                f"Affected modules: {layer_summary}. "
                f"Quantization still proceeded with degraded Hessian inverse "
                f"(pinv=pseudo-inverse, identity=unit matrix). "
                f"Consider checking quantization quality for these modules."
            )
        else:
            logger.info(
                f"[{self._TAG}] Hessian fallback summary: no fallbacks occurred. "
                f"All Cholesky decompositions succeeded."
            )
        self._runtime_model = None
        self._runtime_state_keys.clear()
        self._runtime_device = None
        self._inps = None
        self._outs = None
        self._layer_kwargs = {}
        self._next_expected_layer_index = None
        self._calib_batch_size = 1
        self._prev_topk_indices = None
        self._mtp_embeds = None
        if self._model_obj is not None and hasattr(self._model_obj, "release_empty_model"):
            self._model_obj.release_empty_model()
        bh.empty_cache()
        logger.info(f"[{self._TAG}] finish")

    def _build_runtime_model(self) -> nn.Module:
        return self._model_obj.prepare_empty_model()

    def _record_fallback(self, handler, layer_name: str) -> None:
        """Record Hessian fallback event if handler._hinv_fallback is set."""
        fb = getattr(handler, "_hinv_fallback", None)
        if fb is not None:
            self._fallback_events.append((layer_name, fb))

    _EXPERT_TENSOR_RE = re.compile(r"^(.+)\.experts\.(\d+)\.([^.]+)\.(.+)$")
    _EXPERT_FUSED_RE = re.compile(r"^(.+)\.experts\.([^.]+)(?:\.([^.]+))?$")

    def _fuse_expert_tensors(self, layer) -> None:
        """Fuse expanded per-expert checkpoint tensors into 3D format in layer.tensors.

        Converts experts.0.gate_proj.weight + experts.0.up_proj.weight -> experts.gate_up_proj [E,2I,H]
        and experts.0.down_proj.weight -> experts.down_proj [E,H,I].

        The fused tensor name strips the .weight suffix so it matches the runtime
        model's state dict key (nn.Parameter is stored without .weight suffix).

        Called before _assign_runtime_tensors so that 3D tensor names match
        the 3D Parameters in the original GlmMoeDsaExperts runtime model.
        """
        fusion_map = getattr(self._model_obj, "moe_expert_fusion_map", {})
        if not fusion_map:
            return

        component_to_fused: Dict[str, str] = {}
        for fused_name, (components, _op) in fusion_map.items():
            for comp in components:
                component_to_fused[comp] = fused_name

        collected: Dict[tuple, Dict[int, Dict[str, torch.Tensor]]] = {}
        to_remove: List[str] = []

        for rel_name, tensor in layer.tensors.items():
            match = self._EXPERT_TENSOR_RE.match(rel_name)
            if not match:
                continue
            prefix = match.group(1)
            expert_idx = int(match.group(2))
            component = match.group(3)
            suffix = match.group(4)
            if component not in component_to_fused:
                continue
            fused_name = component_to_fused[component]
            key = (prefix, fused_name, suffix)
            collected.setdefault(key, {}).setdefault(expert_idx, {})[component] = tensor
            to_remove.append(rel_name)

        if not collected:
            return

        for (prefix, fused_name, suffix), experts_dict in collected.items():
            components, op = fusion_map[fused_name]
            expert_slices: List[torch.Tensor] = []
            for e in sorted(experts_dict.keys()):
                comp_dict = experts_dict[e]
                if op == "cat":
                    parts = [comp_dict[c] for c in components if c in comp_dict]
                    if parts:
                        expert_slices.append(torch.cat(parts, dim=0))
                else:
                    for c in components:
                        if c in comp_dict:
                            expert_slices.append(comp_dict[c])
                            break
            if expert_slices:
                # Strip .weight suffix: runtime model state key is
                # "mlp.experts.gate_up_proj", not "mlp.experts.gate_up_proj.weight"
                if suffix == "weight":
                    fused_rel_name = f"{prefix}.experts.{fused_name}"
                else:
                    fused_rel_name = f"{prefix}.experts.{fused_name}.{suffix}"
                layer.tensors[fused_rel_name] = torch.stack(expert_slices, dim=0)

        for rel_name in to_remove:
            layer.tensors.pop(rel_name, None)

    def _unfuse_expert_tensors(self, layer) -> None:
        """Reverse of _fuse_expert_tensors: split unquantized 3D float expert
        tensors back to per-expert 2D format.

        After _process_layer_handlers, if a MoE layer's experts were skipped
        (in ignore_layers) or not quantized, the 3D float tensor created by
        _fuse_expert_tensors remains in layer.tensors. This method splits it
        back to per-expert 2D so vLLM's expert_params_mapping can load it
        (the mapping expects per-expert 2D for non-quantized weights).

        Only processes 3D floating-point tensors (not int32 packed tensors).
        """
        fusion_map = getattr(self._model_obj, "moe_expert_fusion_map", {})
        if not fusion_map:
            return

        to_unfuse = []
        for key in list(layer.tensors.keys()):
            tensor = layer.tensors[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim != 3 or not tensor.is_floating_point():
                continue
            match = self._EXPERT_FUSED_RE.match(key)
            if not match:
                continue
            prefix = match.group(1)
            fused_name = match.group(2)
            if fused_name not in fusion_map:
                continue
            # Only un-fuse original weight tensors (no suffix in key, meaning
            # ".weight" was stripped by _fuse_expert_tensors). Quantized
            # auxiliary tensors (weight_scale, weight_offset, qweight, etc.)
            # have suffixes and must remain 3D packed format.
            if match.group(3) is not None:
                continue
            suffix = "weight"
            to_unfuse.append((key, prefix, fused_name, suffix))

        if not to_unfuse:
            return

        for key, prefix, fused_name, suffix in to_unfuse:
            components, op = fusion_map[fused_name]
            tensor = layer.tensors[key]
            num_experts = tensor.shape[0]
            logger.info(
                f"[{self._TAG}] Un-fusing 3D float tensor: key={key}, "
                f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device}, "
                f"fused_name={fused_name}, op={op}, "
                f"components={components}, num_experts={num_experts}"
            )
            for e in range(num_experts):
                w2d = tensor[e]
                if op == "cat" and len(components) > 1:
                    mid = w2d.shape[0] // 2
                    gate_w = w2d[:mid].clone()
                    up_w = w2d[mid:].clone()
                    gate_key = f"{prefix}.experts.{e}.{components[0]}.{suffix}"
                    up_key = f"{prefix}.experts.{e}.{components[1]}.{suffix}"
                    layer.tensors[gate_key] = gate_w
                    layer.tensors[up_key] = up_w
                    if e == 0:
                        logger.debug(
                            f"[{self._TAG}]   expert 0: {gate_key} shape={list(gate_w.shape)}, "
                            f"{up_key} shape={list(up_w.shape)}"
                        )
                else:
                    w2d_clone = w2d.clone()
                    down_key = f"{prefix}.experts.{e}.{components[0]}.{suffix}"
                    layer.tensors[down_key] = w2d_clone
                    if e == 0:
                        logger.debug(
                            f"[{self._TAG}]   expert 0: {down_key} shape={list(w2d_clone.shape)}"
                        )
            del layer.tensors[key]
            logger.info(
                f"[{self._TAG}] Un-fuse complete: removed 3D key={key}, "
                f"added {num_experts * len(components)} per-expert 2D tensors"
            )

    def _extract_linear_targets(self, layer, skip_names: List[str]):
        """Extract quantization targets from layer tensors.

        For 2D weights: standard Linear targets (module_rel_name, rel_weight_name, bias_name, tensor, bias, False)
        For 3D weights (fused MoE experts): one target per 3D tensor with is_3d=True.
        The 3D target is split into per-expert handlers by _create_handlers.

        Note: 3D tensors are stored without .weight suffix (matching nn.Parameter
        state dict keys), while 2D tensors have .weight suffix.
        """
        targets = []
        for rel_name, tensor in layer.tensors.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_floating_point():
                continue

            if tensor.ndim == 3:
                # 3D fused MoE Parameter (e.g., mlp.experts.gate_up_proj [E, 2I, H])
                # No .weight suffix; rel_name IS the module path
                module_rel_name = rel_name
                full_weight_name = f"{layer.name}.{rel_name}"
                full_module_name = full_weight_name
                if self.should_skip_name(full_weight_name, skip_names):
                    continue
                if self.should_skip_name(full_module_name, skip_names):
                    continue
                targets.append((module_rel_name, rel_name, None, tensor, None, True))
                continue

            if not rel_name.endswith(".weight"):
                continue
            module_rel_name = rel_name[:-7]
            full_weight_name = f"{layer.name}.{rel_name}"
            full_module_name = f"{layer.name}.{module_rel_name}"
            if self.should_skip_name(full_weight_name, skip_names):
                continue
            if self.should_skip_name(full_module_name, skip_names):
                continue
            if tensor.ndim == 2:
                bias_name = f"{module_rel_name}.bias"
                bias = layer.tensors.get(bias_name)
                targets.append((module_rel_name, rel_name, bias_name, tensor, bias, False))
        return targets

    @staticmethod
    def _full_tensor_name(module_name: str, rel_name: str) -> str:
        if rel_name.startswith(f"{module_name}."):
            return rel_name
        return f"{module_name}.{rel_name}"

    def _collect_module_runtime_tensors(
        self,
        module_name: str,
        module_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        tensor_map: Dict[str, torch.Tensor] = {}
        for rel_name, tensor in module_tensors.items():
            if not torch.is_tensor(tensor):
                continue
            full_name = self._full_tensor_name(module_name, rel_name)
            if full_name in self._runtime_state_keys:
                tensor_map[full_name] = tensor
        return tensor_map

    def _resolve_runtime_device(self, chunk) -> torch.device:
        for tensor in chunk.all_tensors().values():
            if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                return tensor.device
        backend_device = getattr(bh, "device", None)
        if isinstance(backend_device, torch.device):
            return backend_device
        return torch.device(bh.default_device_str())

    def _assign_runtime_tensors(self, tensor_map: Dict[str, torch.Tensor]) -> List[str]:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] runtime model is not initialized")
        if self._runtime_device is None:
            raise RuntimeError(f"[{self._TAG}] runtime device is not initialized")

        assigned: List[str] = []
        for full_name, tensor in tensor_map.items():
            # Align meta-tensor dtype with checkpoint dtype to prevent
            # set_module_tensor_to_device from upcasting (e.g. bf16 -> float32),
            # which doubles GPU memory and causes OOM on large MoE layers.
            module_name, leaf_name = full_name.rsplit(".", 1)
            module = _get_child_module(self._runtime_model, module_name)
            old = getattr(module, leaf_name, None)
            if old is not None and old.is_meta and old.dtype != tensor.dtype:
                if leaf_name in module._parameters:
                    module._parameters[leaf_name] = nn.Parameter(
                        torch.empty_like(old, device="meta", dtype=tensor.dtype),
                        requires_grad=False,
                    )
                elif leaf_name in module._buffers:
                    module._buffers[leaf_name] = torch.empty_like(
                        old, device="meta", dtype=tensor.dtype
                    )

            set_module_tensor_to_device(
                self._runtime_model,
                full_name,
                device=self._runtime_device,
                value=tensor if tensor.device == self._runtime_device else tensor.to(self._runtime_device),
            )
            assigned.append(full_name)
        return assigned

    def _unassign_runtime_tensors(self, tensor_names: List[str]) -> None:
        if self._runtime_model is None:
            return
        for full_name in tensor_names:
            module_name, leaf_name = full_name.rsplit(".", 1)
            module = _get_child_module(self._runtime_model, module_name)
            current = getattr(module, leaf_name)
            meta_tensor = torch.empty_like(current, device="meta")
            set_module_tensor_to_device(
                self._runtime_model,
                full_name,
                device="meta",
                value=meta_tensor,
            )
        # Force garbage collection and cache clearing to free GPU memory
        # before loading the next layer's weights (critical for large MoE layers).
        gc.collect()
        bh.empty_cache()

    @staticmethod
    def _iter_calib_batches(calib_data: Any):
        if calib_data is None:
            return
        for batch in calib_data:
            yield batch

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self._runtime_device is None:
            raise RuntimeError(f"[{self._TAG}] runtime device is not initialized")
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self._runtime_device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _sanitize_layer_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(kwargs)
        sanitized["use_cache"] = False
        if "past_key_values" in sanitized:
            sanitized["past_key_values"] = None
        # Remove prev_topk_indices from captured kwargs - we'll inject it
        # manually per-layer to propagate DSA state across streaming layers.
        sanitized.pop("prev_topk_indices", None)
        return sanitized

    @staticmethod
    def _extract_topk_indices(output: Any) -> Optional[torch.Tensor]:
        """Extract topk_indices from a decoder layer output.

        GLM-5's GlmMoeDsaDecoderLayer.forward returns
        (hidden_states, topk_indices). Other models return just hidden_states
        or (hidden_states, ...) without topk_indices.

        Uses a strict check: topk_indices must be an integer tensor (indices,
        not attention weights or other float tensors).
        """
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            second = output[1]
            if torch.is_tensor(second) and not second.is_floating_point():
                return second
        return None

    def _capture_initial_inputs(self, chunk) -> None:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] runtime model is not initialized")
        if not chunk.layers:
            raise ValueError(f"[{self._TAG}] first chunk must contain at least one layer")
        if chunk.calib_data is None:
            raise ValueError(f"[{self._TAG}] calibration data is required")

        class _CaptureInputsStop(RuntimeError):
            pass

        first_layer = _get_child_module(self._runtime_model, chunk.layers[0].name)
        captured: List[torch.Tensor] = []
        layer_kwargs: Dict[str, Any] = {}

        def hook_fn(module, args, kwargs):
            _ = module
            if not args:
                raise _CaptureInputsStop()
            captured.append(args[0].detach())
            if not layer_kwargs:
                for key, value in kwargs.items():
                    if torch.is_tensor(value):
                        layer_kwargs[key] = value.detach()
                    else:
                        layer_kwargs[key] = value
            raise _CaptureInputsStop()

        handle = first_layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        sample_count = 0
        total_batches = len(chunk.calib_data) if hasattr(chunk.calib_data, "__len__") else None
        try:
            with torch.no_grad():
                batch_iter = tqdm(
                    self._iter_calib_batches(chunk.calib_data),
                    total=total_batches,
                    desc=f"{self._TAG.lower()} capture first-layer",
                    leave=True,
                )
                for raw_batch in batch_iter:
                    if not isinstance(raw_batch, dict):
                        continue
                    batch_no_label = dict(raw_batch)
                    batch_no_label.pop("labels", None)
                    batch = self._move_batch_to_device(batch_no_label)
                    try:
                        self._runtime_model(**batch)
                    except _CaptureInputsStop:
                        pass
                    if captured:
                        sample_count += int(captured[-1].shape[0])
                        if sample_count >= self.max_calib_samples:
                            break
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"[{self._TAG}] failed to capture first-layer inputs from calibration data")

        inps = torch.cat(captured, dim=0)
        inps = inps[: self.max_calib_samples]
        self._inps = inps.cpu()
        self._outs = torch.zeros_like(self._inps)
        self._layer_kwargs = self._sanitize_layer_kwargs(layer_kwargs)
        self._calib_batch_size = max(int(captured[0].shape[0]), 1)
        # Save embeddings for MTP layer (inputs_embeds = first layer input)
        if self._quantize_mtp and self._mtp_layer_names:
            self._mtp_embeds = self._inps.clone()

    def _iter_layer_sample_ranges(self, total_samples: int):
        step = max(int(self._calib_batch_size), 1)
        for start in range(0, total_samples, step):
            end = min(start + step, total_samples)
            yield start, end

    def _collect_statistics(
        self,
        layer_module: nn.Module,
        handlers: Dict[str, BaseHessianModule],
        *,
        layer_name: str,
        chunk_index: int,
    ) -> None:
        if self._inps is None:
            raise RuntimeError(f"[{self._TAG}] missing captured inputs")
        hooks = []

        # Separate regular handlers (hook-based) from expert slice handlers.
        # Expert handlers (both per-expert and batched) are NOT in the model's
        # module tree, so forward hooks on them won't fire. Instead, we hook
        # the GlmMoeDsaExperts module and call add_batch manually per expert.
        # expert_handlers_by_module: {experts_path: {global_expert_idx: {proj_type: (handler, local_idx)}}}
        expert_handlers_by_module: Dict[str, Dict[int, Dict[str, tuple]]] = {}

        for handler in handlers.values():
            if getattr(handler, "_is_batched_expert", False):
                # BatchedGPTQModule: covers CH experts
                experts_path = handler._experts_module_path
                proj_type = handler._proj_type
                for local_idx in range(handler._CH):
                    global_idx = handler._expert_start_idx + local_idx
                    expert_handlers_by_module.setdefault(experts_path, {}).setdefault(global_idx, {})[proj_type] = (handler, local_idx)
            elif getattr(handler.layer, "_is_expert_slice", False):
                # Single expert GPTQModule
                layer = handler.layer
                experts_path = layer._experts_module_path
                expert_idx = layer._expert_idx
                proj_type = layer._proj_type
                expert_handlers_by_module.setdefault(experts_path, {}).setdefault(expert_idx, {})[proj_type] = (handler, 0)
            else:
                # Regular 2D Linear handler
                hooks.append(
                    handler.layer.register_forward_hook(
                        lambda module, inp, out, h=handler: h.add_batch(inp[0].data, _unwrap_output(out).data)
                    )
                )

        # Register expert collection hooks on GlmMoeDsaExperts modules
        for experts_path, expert_dict in expert_handlers_by_module.items():
            experts_module = _get_child_module(layer_module, experts_path)
            if experts_module is not None:
                hooks.append(self._register_expert_collection_hook(experts_module, expert_dict))

        total_samples = int(self._inps.shape[0])
        step = max(int(self._calib_batch_size), 1)
        total_steps = (total_samples + step - 1) // step

        with torch.no_grad():
            sample_iter = tqdm(
                self._iter_layer_sample_ranges(total_samples),
                total=total_steps,
                desc=f"{self._TAG.lower()} calib c{chunk_index} {layer_name}",
                leave=True,
                disable=total_steps <= 1,
            )
            for start, end in sample_iter:
                inp = self._inps[start:end].to(self._runtime_device)
                # Inject prev_topk_indices for DSA (GLM-5 shared indexer layers)
                kwargs = dict(self._layer_kwargs)
                if self._prev_topk_indices is not None:
                    kwargs["prev_topk_indices"] = self._prev_topk_indices[start:end].to(self._runtime_device)
                out = layer_module(inp, **kwargs)
                # Track topk_indices for next layer (full indexer layers produce new ones)
                topk = self._extract_topk_indices(out)
                if topk is not None:
                    topk_cpu = topk.detach().cpu()
                    if self._prev_topk_indices is None:
                        # Pre-allocate full tensor with total_samples rows
                        full_shape = (total_samples,) + tuple(topk_cpu.shape[1:])
                        self._prev_topk_indices = torch.zeros(
                            full_shape, dtype=topk_cpu.dtype, device='cpu'
                        )
                    self._prev_topk_indices[start:end] = topk_cpu
        for hook in hooks:
            hook.remove()
        for handler in handlers.values():
            handler.preproc()

        # After Hessian collection, the 3D MoE Parameters (gate_up_proj,
        # down_proj) are no longer needed on GPU for forward passes.
        # Move them to CPU immediately to free ~19 GB of VRAM for the
        # subsequent fasterquant computation.
        #
        # CRITICAL: _ExpertSliceLinear.weight holds a GPU view into the
        # old 3D Parameter data. Replacing w3d.data alone does NOT free
        # the GPU memory because the slice linear views keep it alive.
        # We must also update each slice linear's weight to point to the
        # new CPU data, releasing the old GPU views.
        #
        # All handlers for the same 3D Parameter share the same w3d object.
        # Deduplicate by id(w3d) so we move each 3D Parameter exactly once
        # and update ALL slice linears across ALL handlers.
        w3d_to_slice_linears: Dict[int, List] = {}
        for handler in handlers.values():
            if getattr(handler, "_is_batched_expert", False):
                slice_linears = handler._slice_linears
            elif hasattr(handler, 'layer') and hasattr(handler.layer, '_weight_3d'):
                slice_linears = [handler.layer]
            else:
                continue
            w3d = slice_linears[0]._weight_3d
            if w3d is not None:
                w3d_to_slice_linears.setdefault(id(w3d), (w3d, []))
                w3d_to_slice_linears[id(w3d)][1].extend(slice_linears)

        for w3d, all_slice_linears in w3d_to_slice_linears.values():
            if w3d.data.device.type != "cpu":
                w3d.data = w3d.data.to("cpu")
            # Update ALL slice linears (across all handlers) to release
            # stale GPU views, even if w3d was already moved by a previous handler.
            for sl in all_slice_linears:
                sl.weight = nn.Parameter(
                    w3d.data[sl._expert_idx], requires_grad=False
                )
        bh.empty_cache()

    @staticmethod
    def _register_expert_collection_hook(
        experts_module: nn.Module,
        expert_handlers: Dict[int, Dict[str, tuple]],
    ):
        """Register a forward hook on GlmMoeDsaExperts to capture per-expert inputs.

        The hook intercepts the experts module's forward, extracts routing info
        (hidden_states, top_k_index), and for each hit expert:
        - Calls handler.add_batch_for_expert(x, local_idx) for batched handlers
        - Calls handler.add_batch(x, None) for single-expert handlers
        - Recomputes act(gate)*up for down_proj input
        """
        act_fn = getattr(experts_module, "act_fn", F.silu)
        num_experts = getattr(experts_module, "num_experts", 0)
        gate_up_proj = getattr(experts_module, "gate_up_proj", None)
        # Cache intermediate activations per expert within a single hook call
        # to avoid recomputing F.linear when both gate_up_proj and down_proj
        # handlers exist for the same expert.
        _intermediate_cache: Dict[int, torch.Tensor] = {}

        def hook(module, inp, out):
            if not isinstance(inp, tuple) or len(inp) < 2:
                return
            hidden_states = inp[0]
            top_k_index = inp[1]
            if num_experts <= 0:
                return
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                for expert_idx_tensor in expert_hit:
                    e = int(expert_idx_tensor.item())
                    if e not in expert_handlers:
                        continue
                    pos, tok_idx = torch.where(expert_mask[e])
                    x = hidden_states[tok_idx]
                    handlers_for_e = expert_handlers[e]
                    needs_gate_up = "gate_up_proj" in handlers_for_e
                    needs_down = "down_proj" in handlers_for_e

                    if needs_gate_up and gate_up_proj is not None:
                        h, local_idx = handlers_for_e["gate_up_proj"]
                        if hasattr(h, "add_batch_for_expert"):
                            h.add_batch_for_expert(x, local_idx)
                        else:
                            h.add_batch(x, None)
                    elif needs_gate_up:
                        logger.warning(
                            f"[{self._TAG}] gate_up_proj is None on experts_module, "
                            f"skipping Hessian for expert {e}"
                        )

                    # Only recompute act(gate)*up if down_proj handler exists.
                    # Cache the result to avoid recomputation if multiple
                    # code paths need it within the same expert iteration.
                    if needs_down and gate_up_proj is not None:
                        if e not in _intermediate_cache:
                            gu = F.linear(x, gate_up_proj[e])
                            gate, up = gu.chunk(2, dim=-1)
                            _intermediate_cache[e] = act_fn(gate) * up
                        intermediate = _intermediate_cache[e]
                        h, local_idx = handlers_for_e["down_proj"]
                        if hasattr(h, "add_batch_for_expert"):
                            h.add_batch_for_expert(intermediate, local_idx)
                        else:
                            h.add_batch(intermediate, None)
                    elif needs_down:
                        logger.warning(
                            f"[{self._TAG}] gate_up_proj is None, cannot recompute "
                            f"down_proj input for expert {e}"
                        )

                # Clear cache after processing all experts in this batch
                _intermediate_cache.clear()

        return experts_module.register_forward_hook(hook)

    def _forward_layer_outputs(
        self,
        layer_module: nn.Module,
        *,
        layer_name: str,
        chunk_index: int,
    ) -> None:
        if self._inps is None or self._outs is None:
            raise RuntimeError(f"[{self._TAG}] missing captured inputs")
        total_samples = int(self._inps.shape[0])
        step = max(int(self._calib_batch_size), 1)
        total_steps = (total_samples + step - 1) // step
        with torch.no_grad():
            sample_iter = tqdm(
                self._iter_layer_sample_ranges(total_samples),
                total=total_steps,
                desc=f"{self._TAG.lower()} forward c{chunk_index} {layer_name}",
                leave=True,
                disable=total_steps <= 1,
            )
            for start, end in sample_iter:
                inp = self._inps[start:end].to(self._runtime_device)
                # Inject prev_topk_indices for DSA (GLM-5 shared indexer layers)
                kwargs = dict(self._layer_kwargs)
                if self._prev_topk_indices is not None:
                    kwargs["prev_topk_indices"] = self._prev_topk_indices[start:end].to(self._runtime_device)
                out = layer_module(inp, **kwargs)
                self._outs[start:end] = _unwrap_output(out).detach().cpu()
                # Track topk_indices for next layer (full indexer layers produce new ones)
                topk = self._extract_topk_indices(out)
                if topk is not None:
                    topk_cpu = topk.detach().cpu()
                    if self._prev_topk_indices is None:
                        # Pre-allocate full tensor with total_samples rows
                        full_shape = (total_samples,) + tuple(topk_cpu.shape[1:])
                        self._prev_topk_indices = torch.zeros(
                            full_shape, dtype=topk_cpu.dtype, device='cpu'
                        )
                    self._prev_topk_indices[start:end] = topk_cpu
        self._inps, self._outs = self._outs, self._inps

    def _validate_chunk_order(self, chunk) -> None:
        if not chunk.layers:
            return
        indices = chunk.layer_indices
        if self._next_expected_layer_index is None:
            self._next_expected_layer_index = indices[0]
        expected = self._next_expected_layer_index
        for idx in indices:
            if idx != expected:
                raise ValueError(
                    f"[{self._TAG}] chunk order mismatch: expected layer {expected}, got {idx}"
                )
            expected += 1
        self._next_expected_layer_index = expected

    def process_chunk(self, chunk) -> Any:
        if self._runtime_model is None:
            raise RuntimeError(f"[{self._TAG}] on_start must be called before process_chunk")
        self._validate_chunk_order(chunk)
        if self._runtime_device is None:
            self._runtime_device = self._resolve_runtime_device(chunk)

        skip_names = self._set_skip_from_chunk_metadata(chunk)
        quantized_tensor_names: set[str] = set()
        processed_weights = 0

        if self._inps is None:
            if not chunk.is_first_chunk:
                raise ValueError(
                    f"[{self._TAG}] first processed chunk must include layer-0 to capture initial inputs"
                )
            pre_tensor_map: Dict[str, torch.Tensor] = {}
            for module in chunk.pre_modules:
                pre_tensor_map.update(
                    self._collect_module_runtime_tensors(module.name, module.tensors)
                )
            pre_assigned = self._assign_runtime_tensors(pre_tensor_map)
            try:
                self._capture_initial_inputs(chunk)
            finally:
                self._unassign_runtime_tensors(pre_assigned)

        layer_iter = tqdm(
            chunk.layers,
            total=chunk.layer_count,
            desc=f"{self._TAG.lower()} chunk {chunk.chunk_index}",
            leave=True,
            disable=chunk.layer_count <= 1,
        )
        for layer in layer_iter:
            self._fuse_expert_tensors(layer)
            layer_tensor_map = self._collect_module_runtime_tensors(layer.name, layer.tensors)
            layer_assigned = self._assign_runtime_tensors(layer_tensor_map)
            try:
                layer_module = _get_child_module(self._runtime_model, layer.name)
                targets = self._extract_linear_targets(layer, skip_names)
                handlers = self._create_handlers(layer_module, targets)

                if handlers:
                    self._collect_statistics(
                        layer_module,
                        handlers,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )

                layer_quantized_names, layer_processed_weights = self._process_layer_handlers(
                    layer,
                    targets,
                    handlers,
                    chunk,
                )
                # Un-fuse any 3D float expert tensors that were not quantized
                # (e.g., when the layer is in ignore_layers). The 3D tensor was
                # created by _fuse_expert_tensors for forward propagation but
                # was never quantized, so split it back to per-expert 2D for
                # vLLM compatibility.
                self._unfuse_expert_tensors(layer)
                quantized_tensor_names.update(layer_quantized_names)
                processed_weights += layer_processed_weights

                for handler in handlers.values():
                    handler.free()

                if layer.index < self._total_layers - 1 or self._should_forward_last_for_mtp():
                    # Move 3D MoE parameters back to GPU for forward pass.
                    # They were moved to CPU after _collect_statistics to save
                    # VRAM during fasterquant. Now fasterquant is done and
                    # handlers are freed, so there's room for the 3D params
                    # on GPU. This keeps _forward_layer_outputs fast (GPU MoE
                    # forward vs ~100x slower CPU forward).
                    # Use non_blocking=True for async CPU->GPU transfer.
                    seen_w3d: set = set()
                    for handler in handlers.values():
                        if getattr(handler, "_is_batched_expert", False):
                            w3d = handler._slice_linears[0]._weight_3d
                        elif hasattr(handler, 'layer') and hasattr(handler.layer, '_weight_3d'):
                            w3d = handler.layer._weight_3d
                        else:
                            continue
                        if id(w3d) in seen_w3d:
                            continue
                        seen_w3d.add(id(w3d))
                        if w3d is not None and w3d.data.device.type == "cpu":
                            w3d.data = w3d.data.to(self._runtime_device, non_blocking=True)

                    self._forward_layer_outputs(
                        layer_module,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )
                    # Move 3D MoE parameters back to CPU after forward to free GPU memory
                    # (critical for MTP processing which needs GPU memory for a new layer).
                    # Use non_blocking=True for async GPU->CPU transfer; synchronize
                    # is not needed since the next layer's _assign_runtime_tensors
                    # will allocate new tensors before these are accessed again.
                    seen_w3d_post: set = set()
                    for handler in handlers.values():
                        if getattr(handler, "_is_batched_expert", False):
                            w3d = handler._slice_linears[0]._weight_3d
                        elif hasattr(handler, 'layer') and hasattr(handler.layer, '_weight_3d'):
                            w3d = handler.layer._weight_3d
                        else:
                            continue
                        if w3d is not None and id(w3d) not in seen_w3d_post:
                            seen_w3d_post.add(id(w3d))
                            if w3d.data.device.type != "cpu":
                                w3d.data = w3d.data.to("cpu", non_blocking=True)
                    bh.empty_cache()
            finally:
                self._unassign_runtime_tensors(layer_assigned)

        self._finalize_chunk_metadata(chunk, quantized_tensor_names)
        logger.info(
            f"[{self._TAG}] chunk={chunk.chunk_index}, layers={chunk.layer_count}, "
            f"processed_weights={processed_weights}"
        )
        return chunk

    # === MTP (Multi-Token Prediction) layer support ===

    def _should_forward_last_for_mtp(self) -> bool:
        """Whether to forward the last regular layer to get MTP input."""
        return (self._quantize_mtp or self._save_mtp_debug) and bool(self._mtp_layer_names)

    def process_mtp_chunk(self, chunk) -> Any:
        """Process MTP layer(s) after all regular chunks are done.

        Called by CompressorTask when quantize_mtp=True.
        Expects chunk.layers to contain MTP layer tensors.
        """
        if not self._mtp_layer_names or self._mtp_embeds is None:
            logger.warning(f"[{self._TAG}] MTP processing skipped: no MTP layers or embeddings")
            return chunk

        if self._runtime_device is None:
            self._runtime_device = self._resolve_runtime_device(chunk)

        # Free GPU memory before MTP processing (regular layer 3D params may still be cached)
        bh.empty_cache()

        # Save layer 77 output + embeddings for offline debugging
        if os.environ.get("NPUSLIM_SAVE_MTP_DEBUG") == "1" or self._save_mtp_debug:
            self._save_mtp_inputs(".")

        # If not quantizing MTP, keep per-expert 2D FLOAT format (original checkpoint
        # format). vLLM's expert_params_mapping natively supports per-expert 2D for
        # non-quantized weights, so no fusion is needed.
        if not self._quantize_mtp:
            # Mark all tensors as FLOAT for saver
            chunk.metadata["tensor_types"] = {
                name: "FLOAT" for name in chunk.all_tensors().keys()
            }
            logger.info(
                f"[{self._TAG}] MTP quantization disabled, "
                f"keeping per-expert 2D FLOAT format, debug inputs saved"
            )
            return chunk
        # Auto-detect: if fused 3D expert parameters won't fit on GPU, switch to CPU.
        # At this point tensors are per-expert (2D), but fusion creates 3D params
        # that are ~24 GiB for GLM-5 (256 experts). Estimate from per-expert shapes.
        _mtp_cpu_mode = False
        if self._runtime_device.type == "cuda":
            try:
                gpu_free = torch.cuda.mem_get_info(self._runtime_device.index or 0)[0]
                # Estimate fused 3D size from per-expert tensors
                # Each expert has 3 projections; gate_proj+up_proj fuse into gate_up_proj
                expert_sizes = {}  # expert_idx -> total element count
                for layer in chunk.layers:
                    for name, t in layer.tensors.items():
                        if not torch.is_tensor(t) or ".experts." not in name:
                            continue
                        parts = name.split(".")
                        try:
                            expert_idx = int(parts[parts.index("experts") + 1])
                        except (ValueError, IndexError):
                            continue
                        expert_sizes[expert_idx] = expert_sizes.get(expert_idx, 0) + t.numel()
                n_experts = len(expert_sizes)
                per_expert_elems = sum(expert_sizes.values()) // n_experts if n_experts > 0 else 0
                # fused 3D: ~same total elements as per-expert × 2 bytes (bf16)
                mtp_3d_size = per_expert_elems * n_experts * 2 if n_experts > 0 else 0
                if mtp_3d_size > 0 and gpu_free < mtp_3d_size * 1.2:  # 20% headroom
                    logger.info(
                        f"[{self._TAG}] GPU free={gpu_free / 1e9:.1f} GB, "
                        f"3D experts={mtp_3d_size / 1e9:.1f} GB -> switching to CPU for MTP"
                    )
                    self._mtp_gpu_device = self._runtime_device
                    self._runtime_device = torch.device("cpu")
                    for k, v in self._layer_kwargs.items():
                        if torch.is_tensor(v):
                            self._layer_kwargs[k] = v.to("cpu")
                        elif isinstance(v, (tuple, list)) and all(torch.is_tensor(t) for t in v):
                            self._layer_kwargs[k] = tuple(t.to("cpu") for t in v)
                    _mtp_cpu_mode = True
            except Exception:
                pass  # If detection fails, continue with GPU

        skip_names = list(chunk.metadata.get("skip_layer_names", []))
        # MTP-specific modules that should not be quantized
        mtp_skip = list(getattr(self._model_obj, "mtp_extra_module_names", []))
        skip_names.extend(mtp_skip)

        quantized_tensor_names: set[str] = set()
        processed_weights = 0

        mtp_iter = tqdm(
            chunk.layers,
            total=chunk.layer_count,
            desc=f"{self._TAG.lower()} MTP",
            leave=True,
        )
        try:
            for layer in mtp_iter:
                layer_quantized_names, layer_w = self._process_mtp_layer(layer, skip_names, chunk)
                quantized_tensor_names.update(layer_quantized_names)
                processed_weights += layer_w

            self._finalize_chunk_metadata(chunk, quantized_tensor_names)
            logger.info(
                f"[{self._TAG}] MTP chunk: layers={chunk.layer_count}, "
                f"processed_weights={processed_weights}"
            )
        finally:
            # Restore GPU device if we switched to CPU for MTP.
            # This must be in finally so the device is restored even if
            # _process_mtp_layer raises an exception.
            if _mtp_cpu_mode:
                self._runtime_device = self._mtp_gpu_device
                logger.info(f"[{self._TAG}] Restored runtime device to {self._runtime_device}")

        return chunk

    def _process_mtp_layer(
        self, layer, skip_names: List[str], chunk
    ) -> tuple[set[str], int]:
        """Quantize a single MTP layer."""
        from accelerate.utils import set_module_tensor_to_device

        # 1. Build MtpLayer runtime module (meta device, weights injected later)
        mtp_module = self._build_mtp_module(layer.name)
        mtp_module.eval()

        # 2. Fuse expert tensors (per-expert -> 3D) and inject weights
        self._fuse_expert_tensors(layer)

        # Map checkpoint tensor names to MtpLayer state dict keys
        mtp_block_prefix = "mtp_block."
        mtp_state_keys = set(mtp_module.state_dict().keys())

        tensor_map: Dict[str, torch.Tensor] = {}
        for rel_name, tensor in layer.tensors.items():
            # Direct match (eh_proj, enorm, hnorm, post_norm)
            if rel_name in mtp_state_keys:
                tensor_map[rel_name] = tensor
            # mtp_block sub-modules (input_layernorm, self_attn, mlp, etc.)
            elif f"{mtp_block_prefix}{rel_name}" in mtp_state_keys:
                tensor_map[f"{mtp_block_prefix}{rel_name}"] = tensor
            else:
                # shared_head or other non-MtpLayer tensors - keep as FLOAT
                pass

        # Inject weights into MtpLayer
        # 3D expert tensors go to CPU (loaded on-demand by BatchedGPTQModule),
        # other weights go to GPU for forward pass
        assigned: List[str] = []
        for full_name, tensor in tensor_map.items():
            if tensor.ndim == 3:
                # Expert 3D Parameter - keep on CPU to avoid OOM
                set_module_tensor_to_device(mtp_module, full_name, device="cpu", value=tensor)
            else:
                if tensor.device != self._runtime_device:
                    tensor = tensor.to(self._runtime_device)
                set_module_tensor_to_device(mtp_module, full_name, device=self._runtime_device, value=tensor)
            assigned.append(full_name)

        try:
            # 3. Extract Linear targets from mtp_block + eh_proj
            targets = self._extract_mtp_targets(layer, mtp_module, skip_names)
            handlers = self._create_handlers(mtp_module.mtp_block, targets)

            # Also create handler for eh_proj (if not skipped)
            eh_proj_name = f"{layer.name}.eh_proj"
            if not self.should_skip_name(eh_proj_name, skip_names):
                eh_handler = self._create_eh_proj_handler(mtp_module)
                if eh_handler is not None:
                    handlers["__eh_proj__"] = eh_handler

            # 4. Collect Hessian via MTP forward
            if handlers:
                self._collect_mtp_statistics(mtp_module, handlers, layer_name=layer.name, chunk_index=chunk.chunk_index)

            # 5. Quantize and pack (reuse existing _process_layer_handlers for mtp_block)
            result = self._process_layer_handlers(layer, targets, handlers, chunk)
            if result is None:
                raise RuntimeError(
                    f"[{self._TAG}] _process_layer_handlers returned None for MTP layer {layer.name}"
                )
            layer_quantized_names, layer_processed = result

            # Pack eh_proj separately
            eh_handler = handlers.get("__eh_proj__")
            if eh_handler is not None:
                eh_names, eh_w = self._pack_eh_proj(layer, mtp_module, eh_handler)
                layer_quantized_names.update(eh_names)
                layer_processed += eh_w

            for handler in handlers.values():
                handler.free()

            return layer_quantized_names, layer_processed
        except Exception as e:
            logger.error(f"[{self._TAG}] MTP layer processing failed: {e}")
            raise
        finally:
            del mtp_module
            bh.empty_cache()

    def _build_mtp_module(self, layer_name: str) -> nn.Module:
        """Build an MtpLayer runtime module using transformers' MtpLayer."""
        from accelerate import init_empty_weights
        from transformers.modeling_layers import MtpLayer

        # Get decoder layer class and norm class from the runtime model
        base_model = self._runtime_model
        layer_container = _get_child_module(base_model, self._block_name)
        decoder_layer_cls = type(layer_container[0])
        # Find norm class from the decoder layer
        norm_cls = type(layer_container[0].input_layernorm)

        config = getattr(self._model_obj, "config", None)
        if config is None:
            raise RuntimeError(f"[{self._TAG}] config is required to build MTP module")

        # Determine layer index and use_post_norm
        layer_idx = int(layer_name.split(".")[-1])

        # Extend config lists if MTP layer index is beyond current length
        # (GlmMoeDsaDecoderLayer accesses config.indexer_types[layer_idx] etc.)
        # Use a shallow copy to avoid modifying the original config (which would break config.json save)
        import copy as _copy
        config = _copy.copy(config)
        for list_attr in ("indexer_types", "layer_types", "mlp_layer_types"):
            cfg_list = getattr(config, list_attr, None)
            if isinstance(cfg_list, list) and layer_idx >= len(cfg_list):
                cfg_list = list(cfg_list)  # shallow copy the list
                last_val = cfg_list[-1] if cfg_list else "full"
                while len(cfg_list) <= layer_idx:
                    cfg_list.append(last_val)
                setattr(config, list_attr, cfg_list)

        # Disable post_norm: GlmMoeDsaDecoderLayer returns (hidden_states, topk_indices)
        # tuple, which MtpLayer's post_norm can't handle. post_norm is not needed
        # for Hessian collection.
        use_post_norm = False

        with init_empty_weights():
            mtp = MtpLayer(
                config=config,
                decoder_layer_cls=decoder_layer_cls,
                norm_cls=norm_cls,
                layer_idx=layer_idx,
                use_post_norm=use_post_norm,
            )
        mtp.eval()
        return mtp

    def _extract_mtp_targets(self, layer, mtp_module, skip_names: List[str]):
        """Extract Linear quantization targets from MTP layer.

        Returns targets for mtp_block's Linears (same format as regular layers).
        eh_proj is handled separately.
        """
        # Reuse _extract_linear_targets but with mtp_block's tensors
        # Create a temporary layer-like object for mtp_block
        from npuslim.tasks.compressor.context import LayerInfo

        mtp_block = mtp_module.mtp_block
        mtp_block_name = f"{layer.name}"

        # Collect mtp_block tensors from layer.tensors (without mtp_block. prefix)
        block_tensors: Dict[str, torch.Tensor] = {}
        mtp_block_prefix = "mtp_block."
        mtp_state = mtp_block.state_dict()
        for rel_name in layer.tensors:
            # Check if this tensor belongs to mtp_block
            if rel_name in mtp_state:
                block_tensors[rel_name] = layer.tensors[rel_name]
            elif f"{mtp_block_prefix}{rel_name}" in mtp_state:
                block_tensors[rel_name] = layer.tensors[rel_name]

        temp_layer = LayerInfo(name=mtp_block_name, index=layer.index, tensors=block_tensors)
        return self._extract_linear_targets(temp_layer, skip_names)

    def _create_eh_proj_handler(self, mtp_module) -> Optional[Any]:
        """Create a handler for eh_proj Linear.

        Base class does not know how to create a quantization handler.
        Subclasses (e.g. GPTQAlgorithm) must override this to provide
        an algorithm-specific handler. If eh_proj is not a Linear, returns None.
        """
        eh_proj = mtp_module.eh_proj
        if not isinstance(eh_proj, nn.Linear):
            return None
        raise NotImplementedError(
            f"[{self._TAG}] _create_eh_proj_handler must be overridden in subclass "
            f"to quantize eh_proj. Set quantize_mtp=False or add eh_proj to "
            f"ignore_layers to skip."
        )

    def _pack_eh_proj(self, layer, mtp_module, eh_handler) -> tuple[set[str], int]:
        """Pack quantized eh_proj weights. Override in subclasses."""
        return set(), 0

    def _collect_mtp_statistics(
        self,
        mtp_module: nn.Module,
        handlers: Dict[str, Any],
        *,
        layer_name: str,
        chunk_index: int,
    ) -> None:
        """Collect Hessian statistics for MTP layer via MTP forward pass.

        Handles both regular Linear handlers and MoE expert handlers,
        mirroring the logic in _collect_statistics but using MTP forward.
        """
        if self._inps is None or self._mtp_embeds is None:
            raise RuntimeError(f"[{self._TAG}] missing inputs for MTP statistics")

        hooks = []

        # Separate expert handlers from regular handlers
        expert_handlers_by_module: Dict[str, Dict[int, Dict[str, tuple]]] = {}
        for handler in handlers.values():
            if getattr(handler, "_is_batched_expert", False):
                experts_path = handler._experts_module_path
                proj_type = handler._proj_type
                for local_idx in range(handler._CH):
                    global_idx = handler._expert_start_idx + local_idx
                    expert_handlers_by_module.setdefault(experts_path, {}).setdefault(global_idx, {})[proj_type] = (handler, local_idx)
            elif getattr(handler.layer, "_is_expert_slice", False):
                layer_h = handler.layer
                experts_path = layer_h._experts_module_path
                expert_idx = layer_h._expert_idx
                proj_type = layer_h._proj_type
                expert_handlers_by_module.setdefault(experts_path, {}).setdefault(expert_idx, {})[proj_type] = (handler, 0)
            elif handler is handlers.get("__eh_proj__"):
                # eh_proj hook
                hooks.append(
                    mtp_module.eh_proj.register_forward_hook(
                        lambda module, inp, out, h=handler: h.add_batch(inp[0].data, out.data)
                    )
                )
            else:
                # Regular 2D Linear handler
                target_layer = handler.layer
                hooks.append(
                    target_layer.register_forward_hook(
                        lambda module, inp, out, h=handler: h.add_batch(inp[0].data, _unwrap_output(out).data)
                    )
                )

        # Register expert collection hooks on GlmMoeDsaExperts modules
        for experts_path, expert_dict in expert_handlers_by_module.items():
            # MTP layer's experts are under mtp_block.{experts_path}
            full_path = f"mtp_block.{experts_path}"
            experts_module = _get_child_module(mtp_module, full_path)
            if experts_module is not None:
                hooks.append(self._register_expert_collection_hook(experts_module, expert_dict))
            else:
                logger.warning(f"[{self._TAG}] MTP experts module not found: {full_path}")

        total_samples = int(self._inps.shape[0])
        step = max(int(self._calib_batch_size), 1)
        total_steps = (total_samples + step - 1) // step

        # Move 3D MoE parameters to GPU for forward pass (same as regular layers)
        seen_w3d_mtp: set = set()
        for handler in handlers.values():
            if getattr(handler, "_is_batched_expert", False):
                w3d = handler._slice_linears[0]._weight_3d
            elif hasattr(handler, 'layer') and hasattr(handler.layer, '_weight_3d'):
                w3d = handler.layer._weight_3d
            else:
                continue
            if w3d is not None and id(w3d) not in seen_w3d_mtp:
                seen_w3d_mtp.add(id(w3d))
                if w3d.data.device.type == "cpu":
                    w3d.data = w3d.data.to(self._runtime_device)

        with torch.no_grad():
            sample_iter = tqdm(
                self._iter_layer_sample_ranges(total_samples),
                total=total_steps,
                desc=f"{self._TAG.lower()} mtp calib {layer_name}",
                leave=True,
                disable=total_steps <= 1,
            )
            for start, end in sample_iter:
                inputs_embeds = self._mtp_embeds[start:end].to(self._runtime_device)
                prev_hidden = self._inps[start:end].to(self._runtime_device)

                kwargs = dict(self._layer_kwargs)
                if self._prev_topk_indices is not None:
                    kwargs["prev_topk_indices"] = self._prev_topk_indices[start:end].to(self._runtime_device)

                mtp_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ("position_ids", "attention_mask", "past_key_values",
                                            "use_cache", "position_embeddings")}
                pos_emb = kwargs.get("position_embeddings")
                if pos_emb is not None and isinstance(pos_emb, (tuple, list)):
                    pos_emb = tuple(p.to(self._runtime_device) for p in pos_emb)
                out = mtp_module(
                    inputs_embeds=inputs_embeds,
                    previous_hidden_state=prev_hidden,
                    position_embeddings=pos_emb,
                    attention_mask=None,
                    position_ids=kwargs.get("position_ids"),
                    past_key_values=None,
                    **mtp_kwargs,
                )

        for hook in hooks:
            hook.remove()
        for handler in handlers.values():
            handler.preproc()

        # Move 3D MoE parameters to CPU after Hessian collection (same as regular layers)
        w3d_to_slice_linears: Dict[int, List] = {}
        for handler in handlers.values():
            if getattr(handler, "_is_batched_expert", False):
                slice_linears = handler._slice_linears
            elif hasattr(handler, 'layer') and hasattr(handler.layer, '_weight_3d'):
                slice_linears = [handler.layer]
            else:
                continue
            w3d = slice_linears[0]._weight_3d
            if w3d is not None:
                w3d_to_slice_linears.setdefault(id(w3d), (w3d, []))
                w3d_to_slice_linears[id(w3d)][1].extend(slice_linears)

        for w3d, all_slice_linears in w3d_to_slice_linears.values():
            if w3d.data.device.type != "cpu":
                w3d.data = w3d.data.to("cpu")
            for sl in all_slice_linears:
                sl.weight = nn.Parameter(w3d.data[sl._expert_idx], requires_grad=False)
        bh.empty_cache()

    def _save_mtp_inputs(self, output_dir: str) -> None:
        """Save layer 77 output and embeddings for offline MTP debugging."""
        import os
        from safetensors.torch import save_file

        save_path = os.path.join(output_dir, "mtp_debug_inputs")
        os.makedirs(save_path, exist_ok=True)

        tensors = {}
        if self._inps is not None:
            tensors["layer_last_output.pt"] = self._inps.cpu()
        if self._mtp_embeds is not None:
            tensors["embeddings.pt"] = self._mtp_embeds.cpu()
        if self._prev_topk_indices is not None:
            tensors["prev_topk_indices.pt"] = self._prev_topk_indices.cpu()

        if tensors:
            # Save as safetensors
            st_tensors = {k.replace(".", "_"): v for k, v in tensors.items()}
            save_file(st_tensors, os.path.join(save_path, "mtp_inputs.safetensors"))
            # Save kwargs metadata
            import json
            kwargs_meta = {k: str(type(v).__name__) for k, v in self._layer_kwargs.items()}
            with open(os.path.join(save_path, "layer_kwargs.json"), "w") as f:
                json.dump(kwargs_meta, f, indent=2)
            logger.info(f"[{self._TAG}] Saved MTP debug inputs to {save_path}")
