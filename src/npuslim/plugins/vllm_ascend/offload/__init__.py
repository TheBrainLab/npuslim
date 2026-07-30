"""NPUSlim Offload Trunk — intelligent weight offloading for Ascend NPU.

Enhances vllm-ascend's NPUPrefetchOffloader with:
- Size-aware layer selection (offload largest layers first)
- Automatic HBM memory budget calculation
- Runtime monitoring and statistics
"""

from npuslim.plugins.vllm_ascend.offload.config import OffloadTrunkConfig
from npuslim.plugins.vllm_ascend.offload.memory_budget import (
    MemoryBudget,
    MemoryBudgetCalculator,
)
from npuslim.plugins.vllm_ascend.offload.monitor import OffloadMonitor
from npuslim.plugins.vllm_ascend.offload.npu_prefetch_offloader import (
    EnhancedNPUPrefetchOffloader,
)
from npuslim.plugins.vllm_ascend.offload.planner import OffloadPlan, OffloadPlanner

__all__ = [
    "EnhancedNPUPrefetchOffloader",
    "MemoryBudget",
    "MemoryBudgetCalculator",
    "OffloadMonitor",
    "OffloadPlan",
    "OffloadPlanner",
    "OffloadTrunkConfig",
]
