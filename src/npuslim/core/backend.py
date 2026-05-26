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
    """Unified backend helper shared by runtime, streaming, and distributed code.

    Two read interfaces:
    - **Capability** (immutable): ``detected_name``, ``has_npu``, ``has_cuda`` —
      reflect what hardware is physically present.  Use for workarounds,
      output-format selection, and plugin registration.
    - **Placement** (mutable via ``use()``): ``name``, ``device``, ``module`` —
      the *active* device that controls where tensors live.  Defaults to the
      best available accelerator but can be overridden at runtime.
    """

    def __init__(self) -> None:
        # Immutable hardware detection
        self._has_npu: bool = hasattr(torch, "npu") and torch.npu.is_available()
        self._has_cuda: bool = torch.cuda.is_available()

        # Cache the auto-detected name (never changes)
        self._detected_name: str = self._auto_detect()

        # Mutable active-device state (initialised to auto-detect)
        self._name: str = self._detected_name
        self._device: torch.device = torch.device(self._name)
        self._module = self._resolve_module(self._name)

    # ------------------------------------------------------------------
    # Immutable: hardware capability
    # ------------------------------------------------------------------

    @property
    def detected_name(self) -> str:
        """Auto-detected device name (never changes). Use for capability checks."""
        return self._detected_name

    @property
    def has_npu(self) -> bool:
        """Whether NPU hardware is present (immutable)."""
        return self._has_npu

    @property
    def has_cuda(self) -> bool:
        """Whether CUDA hardware is present (immutable)."""
        return self._has_cuda

    # ------------------------------------------------------------------
    # Mutable: active device for placement
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Active device name. Mutable via ``use()``."""
        return self._name

    @property
    def device(self) -> torch.device:
        """Active torch device. Mutable via ``use()``."""
        return self._device

    @property
    def module(self):
        """Active backend module (``torch.npu`` / ``torch.cuda`` / ``None``). Mutable via ``use()``."""
        return self._module

    def use(self, device_name: str) -> None:
        """Override the active backend for placement decisions.

        Does **not** affect capability queries (``detected_name``,
        ``has_npu``, ``has_cuda``).

        Raises ``ValueError`` for unknown device names.
        Raises ``RuntimeError`` if the requested accelerator is not available.
        """
        normalized = device_name.strip().lower()
        if normalized == "gpu":
            normalized = "cuda"

        if normalized not in ("cpu", "cuda", "npu"):
            raise ValueError(
                f"Unsupported device: {device_name!r}  (choose from cpu/cuda/npu)"
            )
        if normalized == "npu" and not self._has_npu:
            raise RuntimeError("NPU requested but not available on this system")
        if normalized == "cuda" and not self._has_cuda:
            raise RuntimeError("CUDA requested but not available on this system")

        self._name = normalized
        self._device = torch.device(normalized)
        self._module = self._resolve_module(normalized)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect(self) -> str:
        if self._has_npu:
            return "npu"
        if self._has_cuda:
            return "cuda"
        return "cpu"

    @staticmethod
    def _resolve_module(name: str):
        if name == "npu" and hasattr(torch, "npu"):
            return torch.npu
        if name == "cuda":
            return torch.cuda
        return None

    # ------------------------------------------------------------------
    # Device helpers (all driven by the mutable active device)
    # ------------------------------------------------------------------

    def default_device_str(self) -> str:
        if self._name == "npu":
            return "npu:0"
        if self._name == "cuda":
            return "cuda:0"
        return "cpu"

    def device_for_rank(self, local_rank: int) -> str:
        if self._name == "npu":
            return f"npu:{local_rank}"
        if self._name == "cuda":
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

    # ------------------------------------------------------------------
    # Runtime operations
    # ------------------------------------------------------------------

    def sync(self, device: str | None = None) -> None:
        target = (device or self.default_device_str()).lower()
        if target.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
            return
        if target.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def set_device(self, local_rank: int) -> None:
        if self._name == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(local_rank)
            return
        if self._name == "cuda" and torch.cuda.is_available():
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
