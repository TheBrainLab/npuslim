"""GPTQ-style layer-wise quantization skeleton for the v2 step framework.

This is a structural example (handler + observer + hook flow), not full GPTQ math.
It demonstrates:
1. Per-weight handler lifecycle.
2. Observer/hook-style statistics collection.
3. Layer-wise processing with activation carryover across chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from npuslim.algorithms.base_algo import BaseAlgorithm, step
from npuslim.registry import register_algorithm


@dataclass
class GPTQStepwiseConfig:
    wbits: int = 4
    percdamp: float = 0.01
    seed: int = 0
    max_init_tokens: int = 128


class GPTQObserver:
    """Minimal Hessian-diagonal proxy observer."""

    def __init__(self):
        self.hdiag: Optional[torch.Tensor] = None
        self.nsamples: int = 0

    def observe_input(self, projected_activation: torch.Tensor) -> torch.Tensor:
        batch_diag = projected_activation.pow(2).mean(dim=0)
        if self.hdiag is None:
            self.hdiag = batch_diag
            self.nsamples = 1
        else:
            n = float(self.nsamples)
            self.hdiag = (self.hdiag * n + batch_diag) / (n + 1.0)
            self.nsamples += 1
        return self.hdiag


class GPTQLayerHandler:
    """Per-weight handler with its own observer/state."""

    def __init__(self, key: str, wbits: int, percdamp: float):
        self.key = key
        self.wbits = int(wbits)
        self.percdamp = float(percdamp)
        self.observer = GPTQObserver()

    def _int_range(self) -> Tuple[int, int]:
        qmin = -(1 << (self.wbits - 1))
        qmax = (1 << (self.wbits - 1)) - 1
        return qmin, qmax

    def quantize(self, weight: torch.Tensor, hdiag: torch.Tensor) -> torch.Tensor:
        qmin, qmax = self._int_range()
        damp = max(float(hdiag.mean().item()), 1e-6) * self.percdamp
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = (max_abs + damp) / max(qmax, 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return (q * scale).to(dtype=weight.dtype)


@register_algorithm("GPTQStepwise", aliases=["GPTQExample", "gptq_stepwise"])
class GPTQStepwiseAlgorithm(BaseAlgorithm):
    """Simplified GPTQ-style layer-wise quantization skeleton."""

    def __init__(
        self,
        wbits: int = 4,
        percdamp: float = 0.01,
        seed: int = 0,
        max_init_tokens: int = 128,
        **kwargs,
    ):
        super().__init__(
            wbits=wbits,
            percdamp=percdamp,
            seed=seed,
            max_init_tokens=max_init_tokens,
            **kwargs,
        )
        self.cfg = GPTQStepwiseConfig(
            wbits=int(wbits),
            percdamp=float(percdamp),
            seed=int(seed),
            max_init_tokens=max(int(max_init_tokens), 1),
        )
        self._handlers: Dict[str, GPTQLayerHandler] = {}
        self._carry_activation: Optional[torch.Tensor] = None

    def on_start(self, context):
        logger.info(
            "[GPTQStepwise] start (skeleton): "
            f"wbits={self.cfg.wbits}, percdamp={self.cfg.percdamp}"
        )
        self._handlers.clear()
        self._carry_activation = None

    def _flatten_activation(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.float()
        return x.reshape(-1, x.shape[-1]).float()

    def _project_activation(self, x: torch.Tensor, in_features: int) -> torch.Tensor:
        flat = self._flatten_activation(x)
        dim = flat.shape[-1]
        if dim == in_features:
            return flat
        if dim > in_features:
            return flat[:, :in_features]
        pad = torch.zeros(flat.shape[0], in_features - dim, device=flat.device, dtype=flat.dtype)
        return torch.cat([flat, pad], dim=1)

    def _infer_first_in_features(self, layers: List[object]) -> int:
        for layer in layers:
            for _, weight in self._iter_layer_weights(layer):
                if weight.ndim == 2:
                    return int(weight.shape[1])
        return 1024

    def _init_activation_seed(
        self,
        calib_data: Optional[Iterable],
        layers: List[object],
        device: torch.device,
    ) -> torch.Tensor:
        in_features = self._infer_first_in_features(layers)
        rows = 32
        if calib_data is not None:
            try:
                first_batch = next(iter(calib_data))
                if isinstance(first_batch, dict) and "input_ids" in first_batch:
                    token_count = int(first_batch["input_ids"].numel())
                    rows = max(1, min(token_count, self.cfg.max_init_tokens))
            except Exception:
                rows = 32

        g = torch.Generator(device=device)
        g.manual_seed(self.cfg.seed)
        return torch.randn(rows, in_features, device=device, generator=g, dtype=torch.float32)

    def _iter_layer_weights(self, layer: object):
        if isinstance(layer, dict):
            tensors = layer.get("tensors", {})
            if isinstance(tensors, dict):
                for name, tensor in tensors.items():
                    if (
                        isinstance(tensor, torch.Tensor)
                        and tensor.ndim == 2
                        and name.endswith(".weight")
                        and tensor.is_floating_point()
                    ):
                        yield name, tensor
            return

        if isinstance(layer, nn.Module):
            for sub_name, module in layer.named_modules():
                if isinstance(module, nn.Linear) and module.weight is not None:
                    weight_name = "weight" if sub_name == "" else f"{sub_name}.weight"
                    yield weight_name, module.weight.data

    def _assign_quantized_weight(self, layer: object, weight_name: str, quant_w: torch.Tensor) -> None:
        if isinstance(layer, dict):
            tensors = layer.get("tensors", {})
            if isinstance(tensors, dict) and weight_name in tensors:
                tensors[weight_name] = quant_w
            return

        if isinstance(layer, nn.Module):
            module_path = weight_name.removesuffix(".weight")
            current = layer
            if module_path:
                for part in module_path.split("."):
                    current = getattr(current, part)
            with torch.no_grad():
                current.weight.copy_(quant_w)

    def _propagate_activation(self, activation: torch.Tensor, proxy_weight: torch.Tensor) -> torch.Tensor:
        x = self._project_activation(activation, int(proxy_weight.shape[1]))
        out = x @ proxy_weight.float().t()
        return torch.tanh(out)

    def _get_handler(self, key: str) -> GPTQLayerHandler:
        handler = self._handlers.get(key)
        if handler is None:
            handler = GPTQLayerHandler(
                key=key,
                wbits=self.cfg.wbits,
                percdamp=self.cfg.percdamp,
            )
            self._handlers[key] = handler
        return handler

    def _observer_hook(self, handler: GPTQLayerHandler, projected_act: torch.Tensor) -> torch.Tensor:
        """Observer hook call-site (skeleton equivalent of per-layer forward hook)."""
        return handler.observer.observe_input(projected_act)

    @step(order=0, requires=["calib_data"], produces=["chunk_summary"])
    def process_chunk(self, context, calib_data=None):
        payload = context.current_chunk or {}
        layers = payload.get("layers", [])
        chunk_idx = int(payload.get("index", -1))
        if not layers:
            return {"chunk_summary": {"chunk": chunk_idx, "layers": 0, "quantized_weights": 0}}

        if self._carry_activation is None:
            first_weight = next(iter(self._iter_layer_weights(layers[0])), None)
            device = first_weight[1].device if first_weight is not None else torch.device("cpu")
            self._carry_activation = self._init_activation_seed(calib_data, layers, device=device)

        quantized_weights = 0
        for layer_offset, layer in enumerate(layers):
            layer_idx = layer_offset
            if isinstance(layer, dict):
                layer_idx = int(layer.get("index", layer_offset))
            layer_name = f"layer.{layer_idx}"
            proxy_weight: Optional[torch.Tensor] = None

            for weight_name, weight in self._iter_layer_weights(layer):
                handler_key = f"{layer_name}:{weight_name}"
                handler = self._get_handler(handler_key)

                projected = self._project_activation(self._carry_activation, int(weight.shape[1]))
                hdiag = self._observer_hook(handler, projected)

                quant_w = handler.quantize(weight.float(), hdiag.to(weight.device))
                self._assign_quantized_weight(layer, weight_name, quant_w.to(weight.dtype))
                quantized_weights += 1
                if proxy_weight is None:
                    proxy_weight = quant_w

            if proxy_weight is not None:
                self._carry_activation = self._propagate_activation(
                    self._carry_activation, proxy_weight
                )

        summary = {
            "chunk": chunk_idx,
            "layers": len(layers),
            "quantized_weights": quantized_weights,
            "handlers": len(self._handlers),
        }
        context.set_intermediate("chunk_summary", summary)
        logger.info(
            f"[GPTQStepwise] chunk={chunk_idx}, layers={len(layers)}, "
            f"quantized_weights={quantized_weights}, handlers={len(self._handlers)}"
        )
        return {"chunk_summary": summary}

    def on_finish(self, context):
        _ = context
        logger.info(f"[GPTQStepwise] finish: handlers={len(self._handlers)}")
