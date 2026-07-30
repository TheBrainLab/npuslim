"""Configuration schema and parsing for Offload Trunk."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from npuslim.plugins.logging import patch_logger

_CONFIG_KEY = "npuslim_offload_trunk"
_ENV_PREFIX = "NPUSLIM_OFFLOAD_TRUNK_"


@dataclass
class OffloadTrunkConfig:
    """Configuration for NPUSlim Offload Trunk.

    Passed via vllm's ``--additional-config`` under the key
    ``npuslim_offload_trunk``, or via ``NPUSLIM_OFFLOAD_TRUNK_*`` env vars.
    """

    enabled: bool = False

    # === Backend ===
    backend: str = "prefetch"  # "prefetch" | "uva" (uva not yet implemented)

    # === Strategy ===
    strategy: str = "size_aware"  # "group" | "size_aware" | "custom"

    # === Group strategy params ===
    group_size: int = 0
    num_in_group: int = 1
    prefetch_step: int = 1  # max prefetch slots; actual value auto-adapted

    # === Custom strategy params ===
    offload_layer_patterns: List[str] = field(default_factory=list)
    keep_layer_patterns: List[str] = field(default_factory=list)

    # === Param-level filtering ===
    offload_params: Set[str] = field(default_factory=set)

    # === Memory ===
    # Safety margin reserved for estimation error (activations, graph, overhead).
    # KV cache is estimated precisely; this margin covers everything else that
    # vllm's gpu_memory_utilization would otherwise need to account for.
    safety_margin_gb: float = 2.0

    # CPU memory threshold: if offloaded weights exceed this fraction of
    # available CPU memory, abort with an error.
    cpu_memory_threshold: float = 0.6

    # === Strict validation ===
    # When True, abort if the plan's estimated HBM usage exceeds available
    # memory. When False, log a warning but continue (may OOM at runtime).
    strict_memory_check: bool = True

    # === Monitoring ===
    enable_monitor: bool = True
    monitor_log_interval: int = 100

    def validate(self) -> None:
        if self.backend not in ("prefetch", "uva"):
            raise ValueError(
                f"OffloadTrunkConfig.backend must be 'prefetch' or 'uva', got '{self.backend}'"
            )
        if self.strategy not in ("group", "size_aware", "custom"):
            raise ValueError(
                f"OffloadTrunkConfig.strategy must be 'group', 'size_aware', or 'custom', "
                f"got '{self.strategy}'"
            )
        if self.strategy == "group":
            if self.group_size <= 0:
                raise ValueError(
                    "group strategy requires group_size > 0"
                )
            if self.num_in_group > self.group_size:
                raise ValueError(
                    f"num_in_group ({self.num_in_group}) must be <= group_size ({self.group_size})"
                )
        if self.prefetch_step < 1:
            raise ValueError(f"prefetch_step must be >= 1, got {self.prefetch_step}")
        if self.safety_margin_gb < 0:
            raise ValueError("safety_margin_gb must be >= 0")
        if not 0 < self.cpu_memory_threshold <= 1:
            raise ValueError("cpu_memory_threshold must be in (0, 1]")

    def __post_init__(self) -> None:
        if isinstance(self.offload_params, str):
            self.offload_params = {
                s.strip() for s in self.offload_params.split(",") if s.strip()
            }
        if isinstance(self.offload_layer_patterns, str):
            self.offload_layer_patterns = [
                s.strip() for s in self.offload_layer_patterns.split(",") if s.strip()
            ]
        if isinstance(self.keep_layer_patterns, str):
            self.keep_layer_patterns = [
                s.strip() for s in self.keep_layer_patterns.split(",") if s.strip()
            ]


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)


def _parse_float(val: Any) -> float:
    return float(val)


def _parse_int(val: Any) -> int:
    return int(val)


def _parse_set(val: Any) -> Set[str]:
    if isinstance(val, (set, list, tuple)):
        return {str(s).strip() for s in val if str(s).strip()}
    if isinstance(val, str):
        return {s.strip() for s in val.split(",") if s.strip()}
    return set()


def _parse_list(val: Any) -> List[str]:
    if isinstance(val, (list, tuple)):
        return [str(s).strip() for s in val if str(s).strip()]
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return []


def from_additional_config(additional_config: Optional[Dict[str, Any]]) -> OffloadTrunkConfig:
    """Parse OffloadTrunkConfig from vllm's additional_config dict."""
    if not additional_config:
        return _from_env()

    raw = additional_config.get(_CONFIG_KEY)
    if raw is None:
        return _from_env()

    if not isinstance(raw, dict):
        patch_logger.warning(
            f"'{_CONFIG_KEY}' in additional_config must be a dict, got {type(raw).__name__}"
        )
        return _from_env()

    return _from_dict(raw)


def _from_dict(raw: Dict[str, Any]) -> OffloadTrunkConfig:
    config = OffloadTrunkConfig(
        enabled=_parse_bool(raw.get("enabled", False)),
        backend=str(raw.get("backend", "prefetch")),
        strategy=str(raw.get("strategy", "size_aware")),
        group_size=_parse_int(raw.get("group_size", 0)),
        num_in_group=_parse_int(raw.get("num_in_group", 1)),
        prefetch_step=_parse_int(raw.get("prefetch_step", 1)),
        offload_layer_patterns=_parse_list(raw.get("offload_layer_patterns", [])),
        keep_layer_patterns=_parse_list(raw.get("keep_layer_patterns", [])),
        offload_params=_parse_set(raw.get("offload_params", set())),
        safety_margin_gb=_parse_float(raw.get("safety_margin_gb", 2.0)),
        cpu_memory_threshold=_parse_float(raw.get("cpu_memory_threshold", 0.6)),
        strict_memory_check=_parse_bool(raw.get("strict_memory_check", True)),
        enable_monitor=_parse_bool(raw.get("enable_monitor", True)),
        monitor_log_interval=_parse_int(raw.get("monitor_log_interval", 100)),
    )
    config.validate()
    return config


def _from_env() -> OffloadTrunkConfig:
    """Parse OffloadTrunkConfig from environment variables."""
    def _env(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(f"{_ENV_PREFIX}{key}", default)

    enabled_val = _env("ENABLED")
    if enabled_val is None:
        return OffloadTrunkConfig()

    config = OffloadTrunkConfig(
        enabled=_parse_bool(enabled_val),
        backend=_env("BACKEND", "prefetch"),
        strategy=_env("STRATEGY", "size_aware"),
        group_size=_parse_int(_env("GROUP_SIZE", "0")),
        num_in_group=_parse_int(_env("NUM_IN_GROUP", "1")),
        prefetch_step=_parse_int(_env("PREFETCH_STEP", "1")),
        offload_layer_patterns=_parse_list(_env("OFFLOAD_LAYER_PATTERNS", "")),
        keep_layer_patterns=_parse_list(_env("KEEP_LAYER_PATTERNS", "")),
        offload_params=_parse_set(_env("OFFLOAD_PARAMS", "")),
        safety_margin_gb=_parse_float(_env("SAFETY_MARGIN_GB", "2.0")),
        cpu_memory_threshold=_parse_float(_env("CPU_MEMORY_THRESHOLD", "0.6")),
        strict_memory_check=_parse_bool(_env("STRICT_MEMORY_CHECK", "true")),
        enable_monitor=_parse_bool(_env("ENABLE_MONITOR", "true")),
        monitor_log_interval=_parse_int(_env("MONITOR_LOG_INTERVAL", "100")),
    )
    config.validate()
    return config


def resolve_from_vllm_config(vllm_config: Any) -> OffloadTrunkConfig:
    """Resolve OffloadTrunkConfig from a VllmConfig object.

    Reads from ``vllm_config.additional_config`` if present.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    return from_additional_config(additional_config if isinstance(additional_config, dict) else None)
