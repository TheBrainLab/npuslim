"""Shared runtime infrastructure for Hessian-based quantization algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
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

    def __init__(self, max_calib_samples: int = 128, **kwargs):
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
        quantized_type = self._ascend_quant_type if bh.name == "npu" else self._quantized_type_label
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

    def on_finish(self) -> None:
        self._update_quantization_metadata()
        self._runtime_model = None
        self._runtime_state_keys.clear()
        self._runtime_device = None
        self._inps = None
        self._outs = None
        self._layer_kwargs = {}
        self._next_expected_layer_index = None
        self._calib_batch_size = 1
        if self._model_obj is not None and hasattr(self._model_obj, "release_empty_model"):
            self._model_obj.release_empty_model()
        bh.empty_cache()
        logger.info(f"[{self._TAG}] finish")

    def _build_runtime_model(self) -> nn.Module:
        if self._model_obj is not None and hasattr(self._model_obj, "prepare_empty_model"):
            model = self._model_obj.prepare_empty_model()
            if model is not None:
                model.eval()
                return model

        if self._model_config is None:
            raise ValueError(f"[{self._TAG}] model_config is required to build empty runtime model")

        auto_model_cls = self._model_obj._resolve_first_available_class(
            self._model_obj.get_model_loader_candidates(),
            kind="model",
        )
        model_kwargs = dict(getattr(self._model_obj, "model_kwargs", {}) or {})
        trust_remote_code = bool(model_kwargs.get("trust_remote_code", False))
        extra_kwargs: Dict[str, Any] = {}
        if "attn_implementation" in model_kwargs:
            extra_kwargs["attn_implementation"] = model_kwargs["attn_implementation"]
        if "torch_dtype" in model_kwargs:
            extra_kwargs["torch_dtype"] = model_kwargs["torch_dtype"]

        with init_empty_weights():
            model = auto_model_cls.from_config(
                self._model_config,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        model.eval()
        return model

    def _extract_linear_targets(self, layer, skip_names: List[str]):
        targets = []
        for rel_name, tensor in layer.tensors.items():
            if not rel_name.endswith(".weight"):
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_floating_point() or tensor.ndim != 2:
                continue
            module_rel_name = rel_name[:-7]
            full_weight_name = f"{layer.name}.{rel_name}"
            full_module_name = f"{layer.name}.{module_rel_name}"
            if self.should_skip_name(full_weight_name, skip_names):
                continue
            if self.should_skip_name(full_module_name, skip_names):
                continue
            bias_name = f"{module_rel_name}.bias"
            bias = layer.tensors.get(bias_name)
            targets.append((module_rel_name, rel_name, bias_name, tensor, bias))
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
        return sanitized

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
        self._inps = inps
        self._outs = torch.zeros_like(inps)
        self._layer_kwargs = self._sanitize_layer_kwargs(layer_kwargs)
        self._calib_batch_size = max(int(captured[0].shape[0]), 1)

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
        for handler in handlers.values():
            hooks.append(
                handler.layer.register_forward_hook(
                    lambda module, inp, out, h=handler: h.add_batch(inp[0].data, _unwrap_output(out).data)
                )
            )
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
                layer_module(self._inps[start:end], **self._layer_kwargs)
        for hook in hooks:
            hook.remove()
        for handler in handlers.values():
            handler.preproc()

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
                out = layer_module(self._inps[start:end], **self._layer_kwargs)
                self._outs[start:end] = _unwrap_output(out).detach()
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
                quantized_tensor_names.update(layer_quantized_names)
                processed_weights += layer_processed_weights

                for handler in handlers.values():
                    handler.free()

                if layer.index < self._total_layers - 1:
                    self._forward_layer_outputs(
                        layer_module,
                        layer_name=layer.name,
                        chunk_index=chunk.chunk_index,
                    )
            finally:
                self._unassign_runtime_tensors(layer_assigned)

        self._finalize_chunk_metadata(chunk, quantized_tensor_names)
        logger.info(
            f"[{self._TAG}] chunk={chunk.chunk_index}, layers={chunk.layer_count}, "
            f"processed_weights={processed_weights}"
        )
        return chunk
