# src/npuslim/tasks/compressor/context.py
"""Context data structures for compressor tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class LayerInfo:
    """One transformer layer payload in a chunk."""

    name: str
    index: int
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    """One non-transformer module payload (pre/post) in a chunk."""

    name: str
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ChunkContext:
    """
    Chunk containing a slice of model layers.

    `layers` is the source of truth. Each LayerInfo has:
    - `name`: fully qualified layer name, e.g. `model.layers.12`
    - `index`: global layer index
    - `tensors`: tensor name -> tensor, where tensor names are relative to the layer

    `pre_modules` / `post_modules` keep non-layer tensors in explicit order.
    """

    chunk_index: int
    layers: List[LayerInfo] = field(default_factory=list)
    pre_modules: List[ModuleInfo] = field(default_factory=list)
    post_modules: List[ModuleInfo] = field(default_factory=list)
    calib_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_first_chunk(self) -> bool:
        """Check if this is the first chunk."""
        return self.chunk_index == 0

    @property
    def layer_indices(self) -> List[int]:
        """Get global layer indices in this chunk."""
        return [layer.index for layer in self.layers]

    @property
    def layer_count(self) -> int:
        """Get number of layers in this chunk."""
        return len(self.layers)

    @property
    def tensor_count(self) -> int:
        """Get number of tensors in this chunk."""
        layer_tensors = sum(len(layer.tensors) for layer in self.layers)
        pre_tensors = sum(len(module.tensors) for module in self.pre_modules)
        post_tensors = sum(len(module.tensors) for module in self.post_modules)
        return pre_tensors + layer_tensors + post_tensors

    @property
    def tensor_names(self) -> List[str]:
        """Get fully qualified tensor names in this chunk."""
        return list(self.all_tensors().keys())

    @property
    def tensors(self) -> Dict[str, torch.Tensor]:
        """Backward-compatible flattened tensor view."""
        return self.all_tensors()

    @property
    def pre_tensors(self) -> Dict[str, torch.Tensor]:
        """Flattened view for pre-transformer module tensors."""
        result: Dict[str, torch.Tensor] = {}
        for module in self.pre_modules:
            module_prefix = f"{module.name}."
            for tensor_name, tensor in module.tensors.items():
                full_name = tensor_name
                if not full_name.startswith(module_prefix):
                    full_name = f"{module_prefix}{tensor_name}"
                result[full_name] = tensor
        return result

    @property
    def post_tensors(self) -> Dict[str, torch.Tensor]:
        """Flattened view for post-transformer module tensors."""
        result: Dict[str, torch.Tensor] = {}
        for module in self.post_modules:
            module_prefix = f"{module.name}."
            for tensor_name, tensor in module.tensors.items():
                full_name = tensor_name
                if not full_name.startswith(module_prefix):
                    full_name = f"{module_prefix}{tensor_name}"
                result[full_name] = tensor
        return result

    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Get a tensor by fully qualified tensor name."""
        return self.all_tensors().get(name)

    def update_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Update/add tensor by fully qualified tensor name if its layer exists."""
        for module in self.pre_modules:
            module_prefix = f"{module.name}."
            if name.startswith(module_prefix):
                module.tensors[name[len(module_prefix):]] = tensor
                return

        for layer in self.layers:
            layer_prefix = f"{layer.name}."
            if name.startswith(layer_prefix):
                layer.tensors[name[len(layer_prefix):]] = tensor
                return

        for module in self.post_modules:
            module_prefix = f"{module.name}."
            if name.startswith(module_prefix):
                module.tensors[name[len(module_prefix):]] = tensor
                return

        raise KeyError(f"Tensor '{name}' does not belong to any loaded module/layer in this chunk")

    def all_tensors(self) -> Dict[str, torch.Tensor]:
        """Get flattened mapping: full_tensor_name -> tensor."""
        result: Dict[str, torch.Tensor] = {}
        result.update(self.pre_tensors)
        for layer in self.layers:
            layer_prefix = f"{layer.name}."
            for tensor_name, tensor in layer.tensors.items():
                full_name = tensor_name
                if not full_name.startswith(layer_prefix):
                    full_name = f"{layer_prefix}{tensor_name}"
                result[full_name] = tensor
        result.update(self.post_tensors)
        return result

    def filter_tensors(self, patterns: List[str]) -> Dict[str, torch.Tensor]:
        """
        Filter tensors by name patterns.

        Args:
            patterns: List of substrings or patterns to match

        Returns:
            Dict of matching tensors
        """
        import re
        result: Dict[str, torch.Tensor] = {}
        for name, tensor in self.all_tensors().items():
            if any(re.search(p, name) for p in patterns):
                result[name] = tensor
        return result

    def filter_by_prefix(self, prefix: str) -> Dict[str, torch.Tensor]:
        """Get all tensors with names starting with prefix."""
        return {k: v for k, v in self.all_tensors().items() if k.startswith(prefix)}
