"""Runtime monitoring for offload trunk."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from npuslim.plugins.logging import patch_logger


@dataclass
class OffloadStats:
    """Accumulated offload statistics."""

    total_forward_steps: int = 0
    total_prefetch_calls: int = 0
    total_offloaded_layers: int = 0
    total_resident_layers: int = 0
    estimated_hbm_usage_gb: float = 0.0
    estimated_cpu_usage_gb: float = 0.0
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def summary(self) -> str:
        return (
            f"OffloadStats: "
            f"steps={self.total_forward_steps}, "
            f"prefetch_calls={self.total_prefetch_calls}, "
            f"offloaded_layers={self.total_offloaded_layers}, "
            f"resident_layers={self.total_resident_layers}, "
            f"est_hbm={self.estimated_hbm_usage_gb:.2f}GB, "
            f"est_cpu={self.estimated_cpu_usage_gb:.2f}GB, "
            f"elapsed={self.elapsed_seconds:.1f}s"
        )


class OffloadMonitor:
    """Runtime monitor for offload trunk operations.

    Records prefetch calls and periodically logs statistics.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = max(int(log_interval), 1)
        self.stats = OffloadStats()

    def record_plan(
        self,
        total_layers: int,
        offloaded_layers: int,
        plan: Optional[Any] = None,
    ) -> None:
        """Record the offload plan summary."""
        self.stats.total_offloaded_layers = offloaded_layers
        self.stats.total_resident_layers = total_layers - offloaded_layers
        if plan is not None:
            self.stats.estimated_hbm_usage_gb = plan.estimated_hbm_usage / 1e9
            self.stats.estimated_cpu_usage_gb = plan.estimated_cpu_usage / 1e9

        patch_logger.info(
            f"[OffloadMonitor] Plan recorded: "
            f"{offloaded_layers}/{total_layers} layers offloaded, "
            f"est_hbm={self.stats.estimated_hbm_usage_gb:.2f}GB, "
            f"est_cpu={self.stats.estimated_cpu_usage_gb:.2f}GB"
        )

    def record_prefetch(self, layer_idx: int) -> None:
        """Record a prefetch event."""
        self.stats.total_prefetch_calls += 1

    def record_forward_step(self) -> None:
        """Record a forward step."""
        self.stats.total_forward_steps += 1
        if self.stats.total_forward_steps % self.log_interval == 0:
            patch_logger.info(f"[OffloadMonitor] {self.stats.summary()}")

    def final_report(self) -> str:
        """Generate a final report string."""
        return self.stats.summary()
