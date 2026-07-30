"""Intelligent offload planner — decides which layers to offload."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.vllm_ascend.offload.config import OffloadTrunkConfig
from npuslim.plugins.vllm_ascend.offload.memory_budget import MemoryBudget


@dataclass
class OffloadPlan:
    """Plan describing which layers to offload and how."""

    offload_layer_indices: Set[int] = field(default_factory=set)
    resident_layer_indices: Set[int] = field(default_factory=set)
    prefetch_step: int = 1
    estimated_hbm_usage: int = 0
    estimated_cpu_usage: int = 0
    strategy: str = "size_aware"
    total_layers: int = 0

    # Decision provenance: "user" (explicit config), "auto" (heuristic), "profiled" (profiling-driven)
    decision_source: str = "auto"

    # Overlap analysis
    resident_gap: int = 1  # min resident layers between consecutive offloaded layers
    can_fully_overlap: bool = True

    # Buffer pool size in HBM (for StaticBufferPool)
    buffer_pool_bytes: int = 0

    def summary(self) -> str:
        offload_count = len(self.offload_layer_indices)
        resident_count = len(self.resident_layer_indices)
        gap_info = f", gap={self.resident_gap}" if self.resident_gap > 1 else ""
        overlap = "✓" if self.can_fully_overlap else "⚠"
        return (
            f"OffloadPlan: "
            f"offload={offload_count}/{offload_count + resident_count} layers, "
            f"prefetch_step={self.prefetch_step}{gap_info}, "
            f"est_hbm={self.estimated_hbm_usage / 1024**3:.2f}GB, "
            f"est_cpu={self.estimated_cpu_usage / 1024**3:.2f}GB, "
            f"buffer={self.buffer_pool_bytes / 1024**3:.2f}GB, "
            f"overlap={overlap}, "
            f"source={self.decision_source}, "
            f"strategy={self.strategy}"
        )


class OffloadPlanner:
    """Plan which transformer layers to offload based on memory budget and strategy."""

    def plan(
        self,
        vllm_config: Any,
        budget: MemoryBudget,
        config: OffloadTrunkConfig,
    ) -> OffloadPlan:
        num_layers = self._get_num_layers(vllm_config)
        if num_layers == 0:
            patch_logger.warning("[OffloadTrunk] No transformer layers found")
            return OffloadPlan()

        if config.strategy == "group":
            plan = self._plan_group(num_layers, budget, config)
        elif config.strategy == "size_aware":
            plan = self._plan_size_aware(num_layers, vllm_config, budget, config)
        elif config.strategy == "custom":
            plan = self._plan_custom(num_layers, config)
        else:
            patch_logger.warning(f"Unknown strategy '{config.strategy}', falling back to size_aware")
            plan = self._plan_size_aware(num_layers, vllm_config, budget, config)

        # Mark decision source: user explicitly configured vs auto
        if config.strategy == "custom":
            plan.decision_source = "user"
        elif config.strategy == "group" and config.group_size > 0:
            plan.decision_source = "user"
        # size_aware is always "auto" now (no offload_ratio to make it "user")

        # Validate plan against memory budget
        self.validate_plan(plan, budget, config)

        # Log the plan and exact offloaded layer indices
        patch_logger.info(f"[OffloadTrunk] {plan.summary()}")
        if plan.offload_layer_indices:
            patch_logger.info(
                f"[OffloadTrunk] Offloaded layer indices: {sorted(plan.offload_layer_indices)}"
            )

        return plan

    def validate_plan(
        self,
        plan: OffloadPlan,
        budget: MemoryBudget,
        config: OffloadTrunkConfig,
    ) -> None:
        """Validate the plan against memory constraints.

        Checks:
        1. Resident weight + buffer pool must fit in available HBM
        2. CPU offload must fit in available DDR (warn only)
        3. Interleaved layout must satisfy overlap requirements

        Raises:
            RuntimeError: if strict_memory_check=True and HBM is insufficient
        """
        if not plan.offload_layer_indices:
            return

        # Use buffer pool estimate from memory budget (consistent pre-plan reservation)
        plan.buffer_pool_bytes = budget.estimated_buffer_pool_bytes

        total_hbm_needed = plan.estimated_hbm_usage + plan.buffer_pool_bytes

        if total_hbm_needed > budget.available_for_weights:
            msg = (
                f"[OffloadTrunk] Memory insufficient! "
                f"HBM need: {total_hbm_needed / 1024**3:.2f} GB "
                f"(weights {plan.estimated_hbm_usage / 1024**3:.2f} GB + "
                f"buffer_pool {plan.buffer_pool_bytes / 1024**3:.2f} GB), "
                f"available: {budget.available_for_weights / 1024**3:.2f} GB, "
                f"deficit: {(total_hbm_needed - budget.available_for_weights) / 1024**3:.2f} GB.\n"
                f"  Suggestion: decrease max_model_len or max_num_seqs to reduce KV cache, "
                f"or decrease prefetch_step (current {config.prefetch_step}), "
                f"or add more NPU cards (TP/PP)"
            )
            if config.strict_memory_check:
                patch_logger.error(msg)
                raise RuntimeError(msg)
            else:
                patch_logger.warning(msg)
        else:
            patch_logger.info(
                f"[OffloadTrunk] Memory validation passed: "
                f"HBM need {total_hbm_needed / 1024**3:.2f} GB "
                f"<= available {budget.available_for_weights / 1024**3:.2f} GB"
            )

    def _plan_group(
        self,
        num_layers: int,
        budget: MemoryBudget,
        config: OffloadTrunkConfig,
    ) -> OffloadPlan:
        """Group strategy: offload last num_in_group layers of every group_size."""
        offload_indices: Set[int] = set()
        for idx in range(num_layers):
            if idx % config.group_size >= config.group_size - config.num_in_group:
                offload_indices.add(idx)

        resident = set(range(num_layers)) - offload_indices
        return OffloadPlan(
            offload_layer_indices=offload_indices,
            resident_layer_indices=resident,
            prefetch_step=config.prefetch_step,
            estimated_hbm_usage=budget.total_weight_bytes_per_card - budget.required_offload_bytes,
            estimated_cpu_usage=budget.required_offload_bytes,
            strategy="group",
            total_layers=num_layers,
        )

    def _plan_size_aware(
        self,
        num_layers: int,
        vllm_config: Any,
        budget: MemoryBudget,
        config: OffloadTrunkConfig,
    ) -> OffloadPlan:
        """Size-aware strategy with compute-prefetch overlap optimization.

        Two-phase approach:
        1. Determine which layers to offload (largest first to minimize count)
        2. Re-map selected layers to an interleaved layout that maximizes
           compute-prefetch overlap

        The interleaved layout ensures that between any two consecutive
        offloaded layers there is at least one resident layer, whose
        compute time provides the overlap window for the H2D prefetch.
        Without interleaving, consecutive offloaded layers have zero
        overlap (copy starts after compute, must finish before next compute).
        """
        layer_sizes = self._estimate_layer_sizes(num_layers, vllm_config)

        # If no offload needed, return all-resident plan
        if not budget.needs_offload:
            patch_logger.info("[OffloadTrunk] No offload needed — all layers fit in HBM")
            return OffloadPlan(
                resident_layer_indices=set(range(num_layers)),
                prefetch_step=config.prefetch_step,
                estimated_hbm_usage=budget.total_weight_bytes_per_card,
                strategy="size_aware",
                total_layers=num_layers,
            )

        # Auto-calculate: offload exactly what doesn't fit
        required_offload = budget.required_offload_bytes

        # Phase 1: Select which layers to offload (by size, largest first)
        sorted_layers = sorted(
            layer_sizes.items(), key=lambda x: x[1], reverse=True
        )
        num_to_offload = 0
        offload_bytes = 0
        for idx, size in sorted_layers:
            if offload_bytes >= required_offload:
                break
            num_to_offload += 1
            offload_bytes += size

        # Phase 2: Interleave offloaded layers with resident layers
        # for maximum compute-prefetch overlap.
        #
        # Instead of offloading layers 0,1,2,...,N-1 (consecutive, no overlap),
        # we offload every K-th layer: 0, K, 2K, 3K, ...
        # where K = ceil(num_layers / num_to_offload).
        #
        # This guarantees at least K-1 resident layers between consecutive
        # offloaded layers, providing K-1 layer-compute times of overlap
        # for each H2D prefetch.
        offload_indices = self._interleave_layers(
            num_layers, num_to_offload, layer_sizes, sorted_layers
        )

        # Recalculate actual offloaded bytes based on interleaved selection
        offload_bytes = sum(layer_sizes[i] for i in offload_indices)

        # Interleaving may change the selection and reduce offload_bytes.
        # Ensure we still offload enough to fit resident + buffer_pool in HBM.
        buffer_pool = budget.estimated_buffer_pool_bytes
        needed_offload = max(0, budget.total_weight_bytes_per_card + buffer_pool
                             - budget.available_for_weights)
        if offload_bytes < needed_offload and len(offload_indices) < num_layers:
            # Add more layers (next largest not yet offloaded) until sufficient
            already = set(offload_indices)
            for idx, _ in sorted_layers:
                if idx not in already:
                    offload_indices.add(idx)
                    offload_bytes += layer_sizes[idx]
                    if offload_bytes >= needed_offload:
                        break

        resident = set(range(num_layers)) - offload_indices
        estimated_hbm = budget.total_weight_bytes_per_card - offload_bytes

        # With interleaved layout, max consecutive offload is 1,
        # so prefetch_step=1 is sufficient for full overlap.
        # But if HBM has surplus, increase prefetch_step for better overlap.
        max_consecutive = self._max_consecutive_offload(offload_indices, num_layers)

        # Auto-adapt prefetch_step: use HBM surplus to allocate more buffer slots
        resident_weight = estimated_hbm
        hbm_surplus = budget.available_for_weights - resident_weight
        if num_to_offload > 0 and hbm_surplus > 0:
            avg_offload_layer_size = offload_bytes / num_to_offload
            max_slots_by_hbm = int(hbm_surplus // avg_offload_layer_size) if avg_offload_layer_size > 0 else 1
            # prefetch_step = min(user max, HBM capacity, num offloaded layers)
            prefetch_step = max(1, min(config.prefetch_step, max_slots_by_hbm, num_to_offload))
        else:
            prefetch_step = max(1, min(config.prefetch_step, max_consecutive))

        plan = OffloadPlan(
            offload_layer_indices=offload_indices,
            resident_layer_indices=resident,
            prefetch_step=prefetch_step,
            estimated_hbm_usage=estimated_hbm,
            estimated_cpu_usage=offload_bytes,
            strategy="size_aware",
            total_layers=num_layers,
        )
        return plan

    def _plan_custom(
        self,
        num_layers: int,
        config: OffloadTrunkConfig,
    ) -> OffloadPlan:
        """Custom strategy: offload layers matching patterns, excluding keep patterns."""
        offload_indices: Set[int] = set()

        for idx in range(num_layers):
            layer_name = f"model.layers.{idx}"
            should_offload = False

            if config.offload_layer_patterns:
                for pattern in config.offload_layer_patterns:
                    if fnmatch.fnmatch(layer_name, pattern) or re.fullmatch(pattern, layer_name):
                        should_offload = True
                        break
            else:
                # If no offload patterns, consider all layers
                should_offload = True

            # Check keep patterns
            if should_offload and config.keep_layer_patterns:
                for pattern in config.keep_layer_patterns:
                    if fnmatch.fnmatch(layer_name, pattern) or re.fullmatch(pattern, layer_name):
                        should_offload = False
                        break

            if should_offload:
                offload_indices.add(idx)

        resident = set(range(num_layers)) - offload_indices
        return OffloadPlan(
            offload_layer_indices=offload_indices,
            resident_layer_indices=resident,
            prefetch_step=config.prefetch_step,
            strategy="custom",
            total_layers=num_layers,
        )

    def _estimate_layer_sizes(
        self,
        num_layers: int,
        vllm_config: Any,
    ) -> Dict[int, int]:
        """Estimate per-card weight size (bytes) for each transformer layer.

        Returns per-card sizes (divided by TP), consistent with
        MemoryBudget's per-card values.
        """
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return {idx: 0 for idx in range(num_layers)}

        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return {idx: 0 for idx in range(num_layers)}

        bytes_per_param = self._get_bytes_per_param(model_config, vllm_config)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1) if parallel_config else 1
        hidden_size = getattr(hf_config, "hidden_size", 0)
        intermediate_size = getattr(hf_config, "intermediate_size", 0)
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)
        n_routed = getattr(hf_config, "n_routed_experts", None) or getattr(hf_config, "num_experts", 0)
        moe_inter = getattr(hf_config, "moe_intermediate_size", intermediate_size)
        n_shared = getattr(hf_config, "n_shared_experts", 0)

        # Attention params per layer
        kv_lora_rank = getattr(hf_config, "kv_lora_rank", 0)
        q_lora_rank = getattr(hf_config, "q_lora_rank", 0)
        qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 0)

        if kv_lora_rank > 0 and q_lora_rank > 0:
            # MLA attention: q_a + q_b + kv_a + kv_b + o_proj + norms
            num_heads = getattr(hf_config, "num_attention_heads", 1)
            qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", 0)
            if qk_nope_head_dim == 0:
                qk_nope_head_dim = max(0, getattr(hf_config, "qk_head_dim", 0) - qk_rope_head_dim)
            v_head_dim = getattr(hf_config, "v_head_dim", qk_nope_head_dim)
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            attn_params = (
                hidden_size * q_lora_rank +                                      # q_a_proj
                q_lora_rank * (num_heads * qk_head_dim) +                        # q_b_proj
                hidden_size * (kv_lora_rank + qk_rope_head_dim) +                # kv_a_proj_with_mqa
                kv_lora_rank * (num_heads * (qk_nope_head_dim + v_head_dim)) +   # kv_b_proj
                (num_heads * v_head_dim) * hidden_size +                         # o_proj
                q_lora_rank +                                                    # q_a_norm
                kv_lora_rank                                                     # kv_a_norm
            )
        else:
            # Standard attention
            num_heads = getattr(hf_config, "num_attention_heads", 1)
            num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
            # Prefer explicit head_dim from config (e.g. Qwen3 sets head_dim=128
            # while hidden_size // num_heads = 64)
            head_dim = getattr(hf_config, "head_dim", 0) or hidden_size // num_heads
            attn_params = (
                hidden_size * (num_heads * head_dim) +      # q_proj
                hidden_size * (num_kv_heads * head_dim) +   # k_proj
                hidden_size * (num_kv_heads * head_dim) +   # v_proj
                (num_heads * head_dim) * hidden_size         # o_proj
            )

        layer_sizes: Dict[int, int] = {}
        for idx in range(num_layers):
            mlp_params: int
            if idx < first_k_dense or n_routed == 0:
                # Dense MLP: gate + up + down
                mlp_params = 3 * hidden_size * intermediate_size
            else:
                # MoE: routed experts + shared experts
                expert_params = n_routed * 3 * hidden_size * moe_inter
                shared_params = 0
                if n_shared > 0:
                    shared_params = n_shared * 3 * hidden_size * moe_inter
                mlp_params = expert_params + shared_params

            # Router gate (small but include for accuracy)
            if n_routed > 0 and idx >= first_k_dense:
                mlp_params += hidden_size * n_routed

            total_params = attn_params + mlp_params
            layer_sizes[idx] = (total_params * bytes_per_param) // tp_size

        return layer_sizes

    def _get_num_layers(self, vllm_config: Any) -> int:
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return 0
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return 0
        return getattr(hf_config, "num_hidden_layers", 0)

    def _get_bytes_per_param(self, model_config: Any, vllm_config: Any) -> int:
        """Determine bytes per parameter based on dtype and quantization.

        For vllm-ascend quantized models, the quantization string is stored in
        model_config.quantization (e.g., "ascend" for W8A8). We also check
        vllm_config.quant_config for standard vllm quantization.
        """
        # Check model_config.quantization (primary location for vllm-ascend)
        quantization = str(getattr(model_config, "quantization", "") or "").lower()
        if "ascend" in quantization or "int8" in quantization or "w8a8" in quantization:
            return 1
        if "w4a8" in quantization or "w4a16" in quantization or "gptq" in quantization:
            return 1
        if "awq" in quantization:
            return 1

        # Check vllm quant_config (for standard vllm quantization)
        quant_config = getattr(vllm_config, "quant_config", None)
        if quant_config is not None:
            quant_method = str(getattr(quant_config, "quant_method", "") or "").lower()
            if "int8" in quant_method or "w8a8" in quant_method or "ascend" in quant_method:
                return 1

        dtype = getattr(model_config, "dtype", None)
        if dtype is not None:
            import torch
            if dtype in (torch.float16, torch.bfloat16):
                return 2
            if dtype == torch.float32:
                return 4
            if dtype == torch.int8:
                return 1
        return 2

    @staticmethod
    def _interleave_layers(
        num_layers: int,
        num_to_offload: int,
        layer_sizes: Dict[int, int],
        sorted_layers: list,
    ) -> Set[int]:
        """Map offload selection to an interleaved layout for overlap.

        Given that we want to offload `num_to_offload` layers out of
        `num_layers`, produce a set of layer indices that are evenly
        spread across the model, maximizing the number of resident
        layers between consecutive offloaded layers.

        For homogeneous models (all layers same size, e.g. pure MoE):
          offload layers 0, K, 2K, ... where K = num_layers // num_to_offload

        For heterogeneous models (mixed Dense/MoE, different sizes):
          prefer to offload the larger layers, but still spread them
          as evenly as possible. We use a greedy approach: sort all
          layers by size descending, then assign each to the offload
          set at the position that maximizes spread.

        Returns:
            Set of layer indices to offload, interleaved with resident layers.
        """
        if num_to_offload <= 0:
            return set()
        if num_to_offload >= num_layers:
            return set(range(num_layers))

        # Check if all layers have the same size (homogeneous)
        sizes = [layer_sizes[i] for i in range(num_layers)]
        is_homogeneous = len(set(sizes)) == 1

        if is_homogeneous:
            # Simple even spacing: offload every K-th layer, offset so
            # that layer 0 is always resident. Use _interleave_selected
            # for consistent spacing with no adjacent offloaded layers.
            dummy_selected = list(range(num_to_offload))
            return OffloadPlanner._interleave_selected(dummy_selected, num_layers)

        # Heterogeneous: prefer larger layers but maintain spread.
        # Instead of just picking the largest layers (which may be
        # consecutive), we interleave the selected layers across the
        # full model to ensure compute-prefetch overlap.
        offload: Set[int] = set()
        # Candidates sorted by size descending
        candidates = [idx for idx, _ in sorted_layers[:num_to_offload]]

        # Interleave: sort candidates by index, then spread them evenly
        # across the model by remapping to an interleaved layout.
        candidates.sort()
        return OffloadPlanner._interleave_selected(candidates, num_layers)

    @staticmethod
    def _interleave_selected(selected: List[int], num_layers: int) -> Set[int]:
        """Remap selected layer indices to an evenly spaced layout.

        Produces a set of ``len(selected)`` layer indices spread evenly
        across ``num_layers`` layers. Layer 0 is always resident. The
        layout ensures resident layers exist between consecutive offloaded
        layers for compute-prefetch overlap.

        For example, 30 layers out of 78 → {2, 5, 8, 11, ...} with
        consistent ~3-layer spacing.
        """
        n = len(selected)
        if n == 0:
            return set()
        if n >= num_layers:
            return set(range(1, num_layers))

        # Evenly sample n positions from [1, num_layers), excluding 0.
        # This is equivalent to placing n points on a ring of size
        # num_layers and rotating so that position 0 is not selected.
        result: Set[int] = set()
        for i in range(n):
            # Place at round(i * num_layers / n) % num_layers, offset
            # by half the spacing so positions are centered in gaps.
            spacing = num_layers / n
            idx = int(round(i * spacing + spacing / 2)) % num_layers
            if idx == 0:
                idx = 1
            # Resolve collisions by shifting forward
            while idx in result:
                idx = (idx + 1) % num_layers
                if idx == 0:
                    idx = 1
            result.add(idx)
        return result

    @staticmethod
    def _optimize_spread(offload_indices: Set[int], num_layers: int) -> Set[int]:
        """Spread offloaded layers to avoid long consecutive runs.

        For consecutive runs > 2, swap middle layers with nearby
        resident layers to improve prefetch overlap.
        """
        if len(offload_indices) <= 1:
            return offload_indices

        sorted_offload = sorted(offload_indices)
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(sorted_offload)):
            if sorted_offload[i] == sorted_offload[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        if max_consecutive <= 2:
            return offload_indices

        resident = set(range(num_layers)) - offload_indices
        result = set(offload_indices)

        sorted_offload = sorted(result)
        i = 0
        while i < len(sorted_offload):
            run_start = sorted_offload[i]
            run_end = run_start
            j = i + 1
            while j < len(sorted_offload) and sorted_offload[j] == run_end + 1:
                run_end = sorted_offload[j]
                j += 1

            run_len = run_end - run_start + 1
            if run_len > 2:
                for mid in range(run_start + 1, run_end, 2):
                    for candidate in [mid - 2, mid + 2, mid - 1, mid + 1,
                                      mid - 3, mid + 3, mid - 4, mid + 4]:
                        if candidate in resident and 0 <= candidate < num_layers:
                            result.discard(mid)
                            result.add(candidate)
                            resident.discard(candidate)
                            resident.add(mid)
                            break
            i = j

        return result

    @staticmethod
    def _max_consecutive_offload(offload_indices: Set[int], num_layers: int) -> int:
        """Calculate maximum number of consecutive offloaded layers."""
        if not offload_indices:
            return 0
        sorted_offload = sorted(offload_indices)
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(sorted_offload)):
            if sorted_offload[i] == sorted_offload[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        return max_consecutive
