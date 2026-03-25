# src/npuslim/tasks/compressor/context.py
"""Context data structures for compressor tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class LayerInfo:
    """Single layer's data."""

    name: str
    index: int
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkContext:
    """Chunk with all context needed for algorithms."""

    chunk_index: int
    layers: List[LayerInfo]
    calib_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_layer(self, idx: int) -> Optional[LayerInfo]:
        """Get layer by index within chunk."""
        if 0 <= idx < len(self.layers):
            return self.layers[idx]
        return None

    def all_tensors(self) -> Dict[str, torch.Tensor]:
        """Flatten all tensors from all layers with qualified names."""
        result: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            for tensor_name, tensor in layer.tensors.items():
                qualified_name = f"{layer.name}.{tensor_name}"
                result[qualified_name] = tensor
        return result

    def update_tensor(self, qualified_name: str, tensor: torch.Tensor) -> None:
        """Update a tensor by qualified name (e.g., 'layers.0.self_attn.q_proj.weight')."""
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return

        layer_prefix, tensor_suffix = parts
        for layer in self.layers:
            if layer.name == layer_prefix:
                layer.tensors[tensor_suffix] = tensor
                return
