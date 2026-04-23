"""INT8 Dynamic quantization algorithm with observer-hook workflow.

This implementation mirrors legacy `compressor/quantizer/int8_dyn` behavior for:
- observer registry selection (`weight_observer` / `w_quant_method`)
- PTQ observer hook container lifecycle
- per-tensor / per-channel / per-group weight quantization
- exporting both quantized `*.weight` and companion `*.weight_scale`
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from loguru import logger

from npuslim.algorithms.quantization.base_quant_algo import BaseQuantizationAlgorithm
from npuslim.core.backend import bh
from npuslim.core import register_algorithm

if TYPE_CHECKING:
    from npuslim.tasks.compressor.context import ChunkContext, LayerInfo


@dataclass
class INT8DynamicConfig:
    """Configuration for INT8 dynamic quantization."""

    wbits: int = 8
    w_quant_method: str = "per-channel"  # per-tensor / per-channel / per-group
    a_quant_method: str = "per-token"
    group_size: int = -1
    weight_observer: Optional[str] = None


class BaseWeightObserver:
    """Minimal base observer compatible with legacy observer usage."""

    def __init__(self, quant_bits: int = 8, group_size: int = -1):
        self.quant_bits = int(quant_bits)
        self.group_size = int(group_size)
        self._scale: Optional[torch.Tensor] = None
        self._step: int = 0
        self._dtype: Optional[torch.dtype] = None

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def scales(self) -> torch.Tensor:
        if self._step == 0 or self._scale is None:
            raise ValueError(
                f"{self.__class__.__name__} scales must observe weight first"
            )
        if self._dtype is not None:
            return self._scale.to(self._dtype)
        return self._scale


class AbsMaxPerTensorWeightObserver(BaseWeightObserver):
    """Per-tensor abs-max observer."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self._dtype is None:
            self._dtype = weight.dtype
        observed = weight.detach().float().abs().amax().clamp_min(1e-7)
        self._scale = observed
        self._step += 1
        return weight


class AbsMaxChannelWiseWeightObserver(BaseWeightObserver):
    """Per-output-channel abs-max observer (legacy default)."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self._dtype is None:
            self._dtype = weight.dtype
        w = weight.detach().float()
        if w.ndim > 2:
            w = w.flatten(1)
        observed = w.abs().amax(dim=1).clamp_min(1e-7)
        self._scale = observed
        self._step += 1
        return weight


class AbsMaxGroupWiseWeightObserver(BaseWeightObserver):
    """Per-group abs-max observer over input dimension."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.group_size <= 0:
            raise ValueError("per-group observer requires group_size > 0")

        if self._dtype is None:
            self._dtype = weight.dtype
        w = weight.detach().float()
        if w.ndim > 2:
            w = w.flatten(1)

        out_features, in_features = w.shape
        groups = math.ceil(in_features / self.group_size)
        scales: List[torch.Tensor] = []
        for idx in range(groups):
            start = idx * self.group_size
            end = min((idx + 1) * self.group_size, in_features)
            group_absmax = (
                w[:, start:end].abs().amax(dim=1, keepdim=True).clamp_min(1e-7)
            )
            scales.append(group_absmax)

        self._scale = torch.cat(scales, dim=1)
        self._step += 1
        return weight


class PTQObserver:
    """Lightweight observer container aligned with legacy PTQObserver semantics."""

    def __init__(self, weight_observer: Optional[BaseWeightObserver] = None):
        self.weight_observer = weight_observer


class PTQObserverHook:
    """Chunk-level observer-hook manager aligned with legacy lifecycle APIs."""

    def __init__(
        self,
        observer_layers: Dict[str, torch.Tensor],
        weight_observer: Optional[Callable[..., BaseWeightObserver]] = None,
    ):
        self.observer_layers = observer_layers
        self.weight_observer_factory = weight_observer
        self.observer_dict: Dict[str, PTQObserver] = {}

    def apply_hook(self) -> None:
        self.observer_dict = {}
        for name, tensor in self.observer_layers.items():
            obs = None
            if self.weight_observer_factory is not None:
                obs = self.weight_observer_factory(tensor)
            self.observer_dict[name] = PTQObserver(weight_observer=obs)

    def remove_hook(self) -> None:
        self.observer_dict = {}

    def post_process(self) -> None:
        return


@torch.no_grad()
def _expand_group_scales(
    scales: torch.Tensor, in_features: int, group_size: int
) -> torch.Tensor:
    """Expand [out, groups] scales to [out, in_features] with tail-group handling."""
    out_features, groups = scales.shape
    expanded = torch.empty(
        (out_features, in_features), device=scales.device, dtype=scales.dtype
    )

    for idx in range(groups):
        start = idx * group_size
        if start >= in_features:
            break
        end = min((idx + 1) * group_size, in_features)
        expanded[:, start:end] = scales[:, idx : idx + 1]

    return expanded


@torch.no_grad()
def quantize_weight_int(
    weight: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 8,
    group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy-style integer quantization returning quantized weight and stored scale."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(weight.shape)}")

    bnt = max((1 << (int(bits) - 1)) - 1, 1)
    w = weight.detach().float().clone()
    s = scales.detach().float().clone()

    if s.ndim == 0:
        s2 = s.view(1, 1)
    elif s.ndim == 1:
        if s.shape[0] == 1:
            s2 = s.view(1, 1)
        elif s.shape[0] == w.shape[0]:
            s2 = s.view(-1, 1)
        else:
            raise ValueError(
                f"Scale shape {tuple(s.shape)} incompatible with weight shape {tuple(w.shape)}"
            )
    elif s.ndim == 2:
        s2 = s
        if s2.shape[0] not in (1, w.shape[0]):
            raise ValueError(
                f"Scale shape {tuple(s2.shape)} incompatible with weight shape {tuple(w.shape)}"
            )
    else:
        raise ValueError(f"Unsupported scale ndim={s.ndim}")

    s2 = s2.clamp_min(1e-7)

    if s2.shape[1] == 1:
        expanded_scale = s2.expand(-1, w.shape[1])
    else:
        if group_size <= 0:
            repeat = math.ceil(w.shape[1] / s2.shape[1])
            expanded_scale = s2.repeat_interleave(repeat, dim=1)[:, : w.shape[1]]
        else:
            expanded_scale = _expand_group_scales(
                s2, in_features=w.shape[1], group_size=group_size
            )

    stored_scale = s2 / float(bnt)
    quant_weight = torch.round(w / (expanded_scale / float(bnt))).clamp(-bnt - 1, bnt)
    return quant_weight, stored_scale


@register_algorithm("INT8Dynamic", aliases=["INT8Dyn", "int8_dyn"])
class INT8DynamicAlgorithm(BaseQuantizationAlgorithm):
    """INT8 dynamic quantization with observer-hook workflow."""

    _TAG = "INT8Dynamic"
    _ASCEND_QUANT_TYPE = "W8A8_DYNAMIC"
    _WEIGHT_OBSERVERS_CLASS = {
        "per-tensor": AbsMaxPerTensorWeightObserver,
        "per-channel": AbsMaxChannelWiseWeightObserver,
        "per-group": AbsMaxGroupWiseWeightObserver,
    }

    def __init__(
        self,
        wbits: int = 8,
        w_quant_method: str = "per-channel",
        a_quant_method: str = "per-token",
        group_size: int = -1,
        weight_observer: Optional[str] = None,
        **kwargs,
    ):
        if "w_bits" in kwargs:
            legacy_w_bits = kwargs.pop("w_bits")
            if int(wbits) != int(INT8DynamicConfig.wbits) and int(legacy_w_bits) != int(
                wbits
            ):
                logger.warning(
                    f"[{self._TAG}] both wbits={wbits} and legacy w_bits={legacy_w_bits} provided; "
                    f"using wbits={wbits}"
                )
            else:
                wbits = int(legacy_w_bits)

        super().__init__(
            wbits=wbits,
            w_quant_method=w_quant_method,
            a_quant_method=a_quant_method,
            group_size=group_size,
            weight_observer=weight_observer,
            **kwargs,
        )
        self.cfg = INT8DynamicConfig(
            wbits=int(wbits),
            w_quant_method=str(w_quant_method),
            a_quant_method=str(a_quant_method),
            group_size=int(group_size),
            weight_observer=weight_observer,
        )

        self.weight_scales_dict: Dict[str, torch.Tensor] = {}
        self.observer_layers: Dict[str, torch.Tensor] = {}
        self.ptq_hook: Optional[PTQObserverHook] = None

    def on_start(self) -> None:
        logger.info(
            f"[{self._TAG}] start: "
            f"wbits={self.cfg.wbits}, w_quant_method={self.cfg.w_quant_method}, "
            f"a_quant_method={self.cfg.a_quant_method}, group_size={self.cfg.group_size}"
        )
        self.weight_scales_dict.clear()
        self.observer_layers.clear()
        self.ptq_hook = None

    def on_finish(self) -> None:
        self._update_quantization_metadata()
        logger.info(
            f"[{self._TAG}] finish: observed={len(self.observer_layers)}, "
            f"scales={len(self.weight_scales_dict)}"
        )

    @staticmethod
    def _extract_strategy(method: str, default: str) -> str:
        matched = re.search(r"per-([a-zA-Z]+)", method or "")
        if matched:
            return matched.group(1).lower()
        return default

    def _update_quantization_metadata(self) -> None:
        model_config = self._model_config
        if model_config is None:
            return

        w_strategy = self._extract_strategy(self.cfg.w_quant_method, "channel")
        a_strategy = self._extract_strategy(self.cfg.a_quant_method, "token")

        if bh.name == "npu":
            model_config.ascend_quant_config = {
                "model_quant_type": self._ASCEND_QUANT_TYPE,
                "group_size": self.cfg.group_size if w_strategy == "group" else -1,
                "quant_layer_types": ["Linear"],
                "include_g_idx": False,
                "has_offset": False,
            }
            # Ascend runtime consumes quant_model_description.json instead.
            if hasattr(model_config, "quantization_config"):
                try:
                    delattr(model_config, "quantization_config")
                except Exception:
                    pass
        else:
            quantization_config = {
                "quant_method": "compressed-tensors",
                "quantization_status": "compressed",
                "format": "int-quantized",
                "ignore": list(self._skip_layer_names),
                "kv_cache_scheme": None,
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": self.cfg.wbits,
                            "strategy": w_strategy,
                            "dynamic": False,
                            "type": "int",
                        },
                        "input_activations": {
                            "num_bits": 8,
                            "strategy": a_strategy,
                            "dynamic": True,
                            "type": "int",
                        },
                        "output_activations": None,
                    }
                },
            }
            model_config.quantization_config = quantization_config

        self._mark_model_quantized()
        logger.info(f"[{self._TAG}] model quantization metadata updated")

    def _resolve_weight_observer_cls(self):
        obs_key = self.cfg.weight_observer or self.cfg.w_quant_method
        observer_cls = self._WEIGHT_OBSERVERS_CLASS.get(obs_key)
        if observer_cls is None:
            raise ValueError(
                f"Weight observer key '{obs_key}' not found. "
                f"Available: {sorted(self._WEIGHT_OBSERVERS_CLASS.keys())}"
            )
        if obs_key == "per-group" and self.cfg.group_size <= 0:
            raise ValueError("w_quant_method='per-group' requires group_size > 0")
        return observer_cls, obs_key

    @staticmethod
    def _make_scale_name(weight_name: str) -> str:
        if weight_name.endswith(".weight"):
            return f"{weight_name[:-7]}.weight_scale"
        return f"{weight_name}_scale"

    def _collect_observer_layers(
        self, chunk: "ChunkContext"
    ) -> Dict[str, torch.Tensor]:
        skip_layer_names = self._set_skip_from_chunk_metadata(chunk)
        observer_layers: Dict[str, torch.Tensor] = {}
        for layer in chunk.layers:
            for rel_name, tensor in layer.tensors.items():
                full_name = f"{layer.name}.{rel_name}"
                if self.should_skip_name(full_name, skip_layer_names):
                    continue
                if not isinstance(tensor, torch.Tensor):
                    continue
                if not tensor.is_floating_point():
                    continue
                if tensor.ndim != 2:
                    continue
                if not rel_name.endswith(".weight"):
                    continue
                observer_layers[full_name] = tensor
        return observer_layers

    def _build_chunk_tensor_types(
        self,
        chunk: "ChunkContext",
        quantized_tensor_names: set[str],
    ) -> Dict[str, str]:
        tensor_types: Dict[str, str] = {
            name: "FLOAT" for name in chunk.all_tensors().keys()
        }
        for name in quantized_tensor_names:
            if name in tensor_types:
                tensor_types[name] = self._ASCEND_QUANT_TYPE
        return tensor_types

    def process_chunk(self, chunk: "ChunkContext") -> "ChunkContext":
        observer_cls, obs_key = self._resolve_weight_observer_cls()
        self.observer_layers = self._collect_observer_layers(chunk)
        quantized_tensor_names: set[str] = set()

        if not self.observer_layers:
            chunk.metadata["tensor_types"] = self._build_chunk_tensor_types(
                chunk, quantized_tensor_names=quantized_tensor_names
            )
            logger.info(f"[{self._TAG}] chunk has no quantizable weights")
            return chunk

        weight_observer_factory = lambda tensor: observer_cls(  # noqa: E731
            quant_bits=self.cfg.wbits,
            group_size=self.cfg.group_size if obs_key == "per-group" else -1,
        )

        self.ptq_hook = PTQObserverHook(
            observer_layers=self.observer_layers,
            weight_observer=weight_observer_factory,
        )
        self.ptq_hook.apply_hook()

        quantized_count = 0
        try:
            for layer in chunk.layers:
                for rel_name, tensor in list(layer.tensors.items()):
                    full_name = f"{layer.name}.{rel_name}"
                    container = self.ptq_hook.observer_dict.get(full_name)
                    if container is None or container.weight_observer is None:
                        continue

                    observer = container.weight_observer
                    observer(tensor)
                    raw_scale = observer.scales()

                    quant_weight, stored_scale = quantize_weight_int(
                        weight=tensor,
                        scales=raw_scale,
                        bits=self.cfg.wbits,
                        group_size=(
                            self.cfg.group_size if obs_key == "per-group" else -1
                        ),
                    )

                    scale_name = self._make_scale_name(rel_name)
                    layer.tensors[rel_name] = quant_weight.to(dtype=tensor.dtype)
                    layer.tensors[scale_name] = stored_scale.to(dtype=tensor.dtype)
                    self.weight_scales_dict[full_name] = stored_scale
                    quantized_tensor_names.add(full_name)
                    quantized_tensor_names.add(f"{layer.name}.{scale_name}")
                    quantized_count += 1
        finally:
            self.ptq_hook.remove_hook()
            self.ptq_hook.post_process()

        chunk.metadata["tensor_types"] = self._build_chunk_tensor_types(
            chunk, quantized_tensor_names=quantized_tensor_names
        )
        logger.info(
            f"[{self._TAG}] chunk={chunk.chunk_index}, "
            f"observer={obs_key}, quantized_weights={quantized_count}"
        )
        return chunk
