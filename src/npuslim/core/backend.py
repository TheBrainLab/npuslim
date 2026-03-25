"""Backend abstraction for CPU/CUDA/NPU utilities.

Responsibilities:
1. Detect active backend and expose normalized device helpers.
2. Resolve config device-map values to concrete runtime device strings.
3. Provide backend-safe sync/cache-cleanup operations used across modules.
"""

from __future__ import annotations

import gc
from typing import Any

import torch


class BackendHandler:
    """Unified backend helper shared by runtime, streaming, and distributed code."""

    def __init__(self):
        if hasattr(torch, "npu") and torch.npu.is_available():
            self.name = "npu"
            self.device = torch.device("npu")
            self.module = torch.npu
        elif torch.cuda.is_available():
            self.name = "cuda"
            self.device = torch.device("cuda")
            self.module = torch.cuda
        else:
            self.name = "cpu"
            self.device = torch.device("cpu")
            self.module = None

    def default_device_str(self) -> str:
        if self.name == "npu":
            return "npu:0"
        if self.name == "cuda":
            return "cuda:0"
        return "cpu"

    def device_for_rank(self, local_rank: int) -> str:
        if self.name == "npu":
            return f"npu:{local_rank}"
        if self.name == "cuda":
            return f"cuda:{local_rank}"
        return "cpu"

    def resolve_device_map(self, device_map: Any, default: str = "cpu") -> str:
        """Resolve a runtime tensor device string from `device_map` values."""
        if device_map is None:
            return default

        if isinstance(device_map, str):
            normalized = device_map.strip().lower()
            if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
                return self.default_device_str()
            if normalized in {"cuda", "gpu"}:
                return "cuda:0"
            if normalized == "npu":
                return "npu:0"
            if normalized in {"cpu", "disk"}:
                return "cpu"
            return device_map

        if isinstance(device_map, int):
            return f"cuda:{device_map}"

        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, int):
                    return f"cuda:{value}"
                if not isinstance(value, str):
                    continue
                normalized = value.strip().lower()
                if normalized in {"cpu", "disk"}:
                    continue
                if normalized == "cuda":
                    return "cuda:0"
                if normalized == "npu":
                    return "npu:0"
                if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
                    return self.default_device_str()
                return value
            return "cpu"

        return default

    def sync(self, device: str | None = None) -> None:
        target = (device or self.default_device_str()).lower()
        if target.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
            return
        if target.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def set_device(self, local_rank: int) -> None:
        if self.name == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(local_rank)
            return
        if self.name == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    def empty_cache(self, device: str | None = None) -> None:
        target = (device or self.default_device_str()).lower()
        if target.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
            return
        if target.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def full_vacuum(self, device: str | None = None) -> None:
        self.sync(device=device)
        gc.collect()
        self.empty_cache(device=device)


bh = BackendHandler()
