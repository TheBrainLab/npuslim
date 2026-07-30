"""Memory budget calculator for offload trunk planning.

Responsibility:
  Calculate how much HBM is available for model weights, by estimating
  KV cache size precisely and subtracting it from vllm's memory budget.

  Formula:
    requested_memory = total_hbm × gpu_memory_utilization
    available_for_weights = requested_memory - estimated_kv_cache - safety_margin
    required_offload = max(0, total_weight_per_card - available_for_weights)

  KV cache is estimated precisely based on attention type (standard / MLA /
  SFA / DSA) and model config. Activations and graph compilation overhead
  are NOT estimated — they are covered by vllm's gpu_memory_utilization and
  the safety_margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.vllm_ascend.offload.config import OffloadTrunkConfig


@dataclass
class MemoryBudget:
    """Per-card memory budget for model weights."""

    total_hbm_bytes: int
    requested_memory_bytes: int  # total_hbm × gpu_memory_utilization
    kv_cache_bytes: int  # estimated KV cache requirement
    safety_margin_bytes: int  # reserved for activations, graph, overhead
    available_for_weights: int  # requested - kv_cache - safety_margin
    total_weight_bytes_per_card: int
    required_offload_bytes: int
    estimated_buffer_pool_bytes: int = 0  # buffer pool HBM reservation

    @property
    def needs_offload(self) -> bool:
        return self.required_offload_bytes > 0

    def summary(self) -> str:
        return (
            f"MemoryBudget: "
            f"total_hbm={self.total_hbm_bytes / 1024**3:.2f}GB, "
            f"requested={self.requested_memory_bytes / 1024**3:.2f}GB, "
            f"kv_cache={self.kv_cache_bytes / 1024**3:.2f}GB, "
            f"safety_margin={self.safety_margin_bytes / 1024**3:.2f}GB, "
            f"available_for_weights={self.available_for_weights / 1024**3:.2f}GB, "
            f"total_weight_per_card={self.total_weight_bytes_per_card / 1024**3:.2f}GB, "
            f"required_offload={self.required_offload_bytes / 1024**3:.2f}GB"
        )


class MemoryBudgetCalculator:
    """Calculate per-card weight memory budget with precise KV cache estimation."""

    def calculate(
        self,
        vllm_config: Any,
        offload_config: OffloadTrunkConfig,
    ) -> MemoryBudget:
        total_hbm = self._get_total_hbm_bytes(vllm_config)
        gpu_mem_util = self._get_gpu_memory_utilization(vllm_config)
        requested_memory = int(total_hbm * gpu_mem_util)

        kv_cache = self._estimate_kv_cache_bytes(vllm_config)
        safety_margin = int(offload_config.safety_margin_gb * 1024**3)
        available_for_weights = requested_memory - kv_cache - safety_margin

        if available_for_weights <= 0:
            msg = (
                f"[OffloadTrunk] HBM available space insufficient for KV cache: "
                f"requested={requested_memory / 1024**3:.2f}GB, "
                f"kv_cache={kv_cache / 1024**3:.2f}GB, "
                f"safety_margin={safety_margin / 1024**3:.2f}GB, "
                f"remaining={available_for_weights / 1024**3:.2f}GB.\n"
                f"  Suggestion: decrease max_model_len (current {getattr(model_config, 'max_model_len', '?')}), "
                f"or decrease max_num_seqs, "
                f"or increase gpu_memory_utilization (current {gpu_mem_util}), "
                f"or add more NPU cards (TP/PP)"
            )
            patch_logger.error(msg)
            raise RuntimeError(msg)

        total_weight = self._estimate_total_weight_bytes_per_card(vllm_config)

        # required_offload must account for both:
        # 1. The weight deficit (weights that don't fit in HBM)
        # 2. The StaticBufferPool (HBM buffer for prefetch, one layer per slot)
        #
        # Buffer pool size ≈ prefetch_step × max_layer_size, matching the actual
        # StaticBufferPool allocation (slot_capacity copies of unique param shapes).
        # Uses the MAX layer size because the planner offloads the largest layers
        # first, and the buffer pool must fit the largest offloaded layer.
        # Only needed when offload is actually required — if all weights fit in
        # HBM, no buffer pool is allocated and no offload is needed.
        base_offload = max(0, total_weight - available_for_weights)
        if base_offload > 0:
            max_layer_size = self._estimate_max_layer_size_bytes_per_card(vllm_config)
            estimated_buffer_pool = max_layer_size * offload_config.prefetch_step
        else:
            estimated_buffer_pool = 0

        required_offload = max(0, total_weight + estimated_buffer_pool - available_for_weights)

        # Check CPU memory
        self._check_cpu_memory(required_offload, offload_config, vllm_config)

        budget = MemoryBudget(
            total_hbm_bytes=total_hbm,
            requested_memory_bytes=requested_memory,
            kv_cache_bytes=kv_cache,
            safety_margin_bytes=safety_margin,
            available_for_weights=available_for_weights,
            total_weight_bytes_per_card=total_weight,
            required_offload_bytes=required_offload,
            estimated_buffer_pool_bytes=estimated_buffer_pool,
        )
        patch_logger.info(f"[OffloadTrunk] {budget.summary()}")
        return budget

    # === KV Cache Estimation ===

    def _estimate_kv_cache_bytes(self, vllm_config: Any) -> int:
        """Estimate total KV cache memory per card.

        Formula (matching vllm-ascend's max_memory_usage_bytes):
          per_layer = ceil(effective_max_model_len / (block_size × compress_ratio)) × page_size_bytes
          total = num_attn_layers × per_layer

        where:
          effective_max_model_len = ceil(max_model_len / (DCP × PCP))
          page_size_bytes = per-block KV cache size (depends on attention type)
          num_attn_layers = num_hidden_layers on this PP rank
        """
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return 0

        cache_config = getattr(vllm_config, "cache_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if cache_config is None or parallel_config is None:
            return 0

        max_model_len = getattr(model_config, "max_model_len", 4096)
        pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        block_size = getattr(cache_config, "block_size", 128)

        num_layers_total = self._get_num_layers(vllm_config)
        num_attn_layers = (num_layers_total + pp_size - 1) // pp_size if pp_size > 1 else num_layers_total

        if dcp_size * pcp_size > 1:
            effective_max_model_len = (max_model_len + dcp_size * pcp_size - 1) // (dcp_size * pcp_size)
        else:
            effective_max_model_len = max_model_len

        compress_ratio = self._get_compress_ratio(model_config)
        page_size_bytes = self._estimate_page_size_bytes(model_config, parallel_config, cache_config)

        if page_size_bytes == 0 or num_attn_layers == 0:
            return 0

        divisor = block_size * compress_ratio if compress_ratio > 1 else block_size
        blocks_per_layer = (effective_max_model_len + divisor - 1) // divisor
        total_kv = num_attn_layers * blocks_per_layer * page_size_bytes

        patch_logger.info(
            f"[OffloadTrunk] KV cache estimate: "
            f"num_attn_layers={num_attn_layers}, "
            f"effective_max_model_len={effective_max_model_len}, "
            f"block_size={block_size}, compress_ratio={compress_ratio}, "
            f"page_size_bytes={page_size_bytes}, "
            f"blocks_per_layer={blocks_per_layer}, "
            f"total={total_kv / 1024**3:.2f}GB"
        )
        return total_kv

    def _get_compress_ratio(self, model_config: Any) -> int:
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None) if hf_config else None
        if compress_ratios and isinstance(compress_ratios, (list, tuple)) and len(compress_ratios) > 0:
            return int(compress_ratios[0])
        return 1

    def _estimate_page_size_bytes(self, model_config: Any, parallel_config: Any, cache_config: Any) -> int:
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            return 0

        block_size = getattr(cache_config, "block_size", 128)
        dtype_bytes = self._get_kv_cache_dtype_bytes(model_config)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1)
        use_mla = getattr(model_config, "use_mla", False)

        if use_mla:
            kv_lora_rank = getattr(hf_config, "kv_lora_rank", 0)
            qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 0)
            has_index_topk = getattr(hf_config, "index_topk", None) is not None
            has_compress = getattr(hf_config, "compress_ratios", None) is not None
            dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)

            if has_index_topk and not has_compress:
                index_head_dim = getattr(hf_config, "index_head_dim", 0)
                head_size = (kv_lora_rank + qk_rope_head_dim) + index_head_dim * dcp_size
                page_size = block_size * 1 * head_size * dtype_bytes
                patch_logger.info(f"[OffloadTrunk] SFA: kv_lora={kv_lora_rank}, qk_rope={qk_rope_head_dim}, index={index_head_dim}, DCP={dcp_size}, head_size={head_size}, page_size={page_size}")
                return page_size
            elif has_compress:
                head_dim = getattr(hf_config, "head_dim", 0)
                index_head_dim = getattr(hf_config, "index_head_dim", 0)
                head_size = head_dim + index_head_dim * dcp_size
                page_size = block_size * 1 * head_size * dtype_bytes
                patch_logger.info(f"[OffloadTrunk] DSA: head_dim={head_dim}, index={index_head_dim}, DCP={dcp_size}, page_size={page_size}")
                return page_size
            else:
                head_size = kv_lora_rank + qk_rope_head_dim
                page_size = block_size * 1 * head_size * dtype_bytes
                patch_logger.info(f"[OffloadTrunk] MLA: kv_lora={kv_lora_rank}, qk_rope={qk_rope_head_dim}, page_size={page_size}")
                return page_size
        else:
            total_num_kv_heads = self._get_total_num_kv_heads(model_config)
            num_kv_heads = max(1, total_num_kv_heads // tp_size)
            head_dim = self._get_head_dim(model_config)
            page_size = block_size * 2 * num_kv_heads * head_dim * dtype_bytes
            patch_logger.info(f"[OffloadTrunk] Standard: num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}")
            return page_size

    def _get_kv_cache_dtype_bytes(self, model_config: Any) -> int:
        """Get dtype size for KV cache."""
        # Default: use model dtype
        dtype = getattr(model_config, "dtype", None)
        if dtype is not None:
            import torch
            if dtype in (torch.float16, torch.bfloat16):
                return 2
            if dtype == torch.float32:
                return 4
            if dtype == torch.int8:
                return 1
        return 2  # Default bf16/fp16

    def _get_total_num_kv_heads(self, model_config: Any) -> int:
        """Get total number of KV heads from model config."""
        # Try model_arch_config first
        arch_config = getattr(model_config, "model_arch_config", None)
        if arch_config is not None:
            total = getattr(arch_config, "total_num_kv_heads", None)
            if total is not None:
                return total

        # Fallback: read from hf_config
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is not None:
            return getattr(hf_config, "num_key_value_heads",
                          getattr(hf_config, "num_attention_heads", 1))
        return 1

    def _get_head_dim(self, model_config: Any) -> int:
        """Get head dimension from model config."""
        # Try model_arch_config
        arch_config = getattr(model_config, "model_arch_config", None)
        if arch_config is not None:
            head_size = getattr(arch_config, "head_size", None)
            if head_size is not None:
                return head_size

        # Try get_head_size method
        get_head_size = getattr(model_config, "get_head_size", None)
        if callable(get_head_size):
            try:
                return get_head_size()
            except Exception:
                pass

        # Fallback: compute from hf_config
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is not None:
            hidden_size = getattr(hf_config, "hidden_size", 0)
            num_heads = getattr(hf_config, "num_attention_heads", 1)
            explicit = getattr(hf_config, "head_dim", 0)
            return explicit or (hidden_size // num_heads if num_heads > 0 else 0)

        return 0

    # === HBM Detection ===

    def _get_total_hbm_bytes(self, vllm_config: Any) -> int:
        """Detect per-card HBM size from NPU hardware."""
        try:
            import torch
            if hasattr(torch, "npu") and torch.npu.is_available():
                local_rank = self._get_local_rank(vllm_config)
                if local_rank is not None:
                    props = torch.npu.get_device_properties(local_rank)
                    return int(props.total_memory)
                props = torch.npu.get_device_properties(0)
                return int(props.total_memory)
        except Exception as e:
            patch_logger.warning(f"Failed to detect NPU HBM size: {e}")

        patch_logger.warning("Using fallback HBM size: 64GB per card")
        return 64 * 1024**3

    def _get_gpu_memory_utilization(self, vllm_config: Any) -> float:
        """Get gpu_memory_utilization from vllm config."""
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is not None:
            return getattr(cache_config, "gpu_memory_utilization", 0.9)
        return 0.9

    def _get_local_rank(self, vllm_config: Any) -> Optional[int]:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if parallel_config is not None:
            return getattr(parallel_config, "local_rank", None)
        return None

    def _get_num_layers(self, vllm_config: Any) -> int:
        """Get number of hidden layers from model config."""
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return 0
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            return 0
        return getattr(hf_config, "num_hidden_layers", 0)

    # === CPU Memory Check ===

    def _check_cpu_memory(
        self,
        required_offload: int,
        config: OffloadTrunkConfig,
        vllm_config: Any = None,
    ) -> None:
        """Check if CPU has enough memory for offloaded weights."""
        if required_offload <= 0:
            return

        try:
            import psutil
            cpu_available = psutil.virtual_memory().available

            # Each TP worker needs its own copy of offloaded weights in CPU
            # memory. Account for total demand across all workers.
            parallel_config = getattr(vllm_config, "parallel_config", None)
            tp_size = getattr(parallel_config, "tensor_parallel_size", 1) if parallel_config else 1
            total_offload_demand = required_offload * tp_size

            cpu_threshold = int(cpu_available * config.cpu_memory_threshold)

            if total_offload_demand > cpu_threshold:
                msg = (
                    f"[OffloadTrunk] CPU memory insufficient: "
                    f"need to offload {required_offload / 1024**3:.2f}GB/card x {tp_size} cards "
                    f"= {total_offload_demand / 1024**3:.2f}GB, "
                    f"CPU available {cpu_available / 1024**3:.2f}GB, "
                    f"threshold {config.cpu_memory_threshold * 100:.0f}% "
                    f"= {cpu_threshold / 1024**3:.2f}GB.\n"
                    f"  Suggestion: decrease max_model_len or max_num_seqs to reduce KV cache, "
                    f"or add more NPU cards, "
                    f"or increase cpu_memory_threshold (current {config.cpu_memory_threshold})"
                )
                if config.strict_memory_check:
                    patch_logger.error(msg)
                    raise RuntimeError(msg)
                else:
                    patch_logger.warning(msg)
            else:
                patch_logger.info(
                    f"[OffloadTrunk] CPU memory check passed: "
                    f"need {required_offload / 1024**3:.2f}GB/card x {tp_size} cards "
                    f"= {total_offload_demand / 1024**3:.2f}GB <= "
                    f"threshold {cpu_threshold / 1024**3:.2f}GB "
                    f"(available {cpu_available / 1024**3:.2f}GB x {config.cpu_memory_threshold * 100:.0f}%)"
                )
        except ImportError:
            patch_logger.warning("psutil not available, skipping CPU memory check")
        except RuntimeError:
            raise
        except Exception as e:
            patch_logger.warning(f"CPU memory check failed: {e}")

    # === Weight Estimation (unchanged from previous version) ===

    def _estimate_total_weight_bytes_per_card(self, vllm_config: Any) -> int:
        """Estimate total weight bytes per card based on model config."""
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            patch_logger.warning("Cannot estimate weight size: model_config not found")
            return 0

        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            patch_logger.warning("Cannot estimate weight size: hf_config not found")
            return 0

        num_params = self._estimate_num_params(hf_config)
        if num_params == 0:
            return 0

        bytes_per_param = self._get_bytes_per_param(model_config, vllm_config)
        total_bytes = num_params * bytes_per_param

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = 1
        if parallel_config is not None:
            tp_size = getattr(parallel_config, "tensor_parallel_size", 1)

        per_card = total_bytes // tp_size
        return per_card

    def _estimate_max_layer_size_bytes_per_card(self, vllm_config: Any) -> int:
        """Estimate the max per-card layer size for buffer pool estimation.

        Computes per-layer weight sizes (distinguishing dense and MoE layers),
        divides by TP, and returns the maximum. The planner offloads the
        largest layers first, so the buffer pool must fit the largest one.
        """
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return 0

        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            return 0

        hidden_size = getattr(hf_config, "hidden_size", 0)
        num_layers = getattr(hf_config, "num_hidden_layers", 0)
        intermediate_size = getattr(hf_config, "intermediate_size", 0)
        if hidden_size == 0 or num_layers == 0:
            return 0

        # Attention params per layer (same for all layers)
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
            head_dim = getattr(hf_config, "head_dim", 0) or hidden_size // num_heads
            attn_params = (
                hidden_size * (num_heads * head_dim) +      # q_proj
                hidden_size * (num_kv_heads * head_dim) +   # k_proj
                hidden_size * (num_kv_heads * head_dim) +   # v_proj
                (num_heads * head_dim) * hidden_size         # o_proj
            )

        # MLP params per layer type — find the max layer
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)
        n_routed = getattr(hf_config, "n_routed_experts", None) or getattr(hf_config, "num_experts", 0)
        moe_inter = getattr(hf_config, "moe_intermediate_size", intermediate_size)
        n_shared = getattr(hf_config, "n_shared_experts", 0)

        max_layer_params = 0
        for idx in range(num_layers):
            if idx < first_k_dense or n_routed == 0:
                # Dense MLP: gate + up + down
                mlp_params = 3 * hidden_size * intermediate_size
            else:
                # MoE: routed experts + shared experts + router gate
                expert_params = n_routed * 3 * hidden_size * moe_inter
                shared_params = n_shared * 3 * hidden_size * moe_inter if n_shared > 0 else 0
                mlp_params = expert_params + shared_params
                if n_routed > 0:
                    mlp_params += hidden_size * n_routed  # router gate

            total_params = attn_params + mlp_params
            if total_params > max_layer_params:
                max_layer_params = total_params

        bytes_per_param = self._get_bytes_per_param(model_config, vllm_config)

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1) if parallel_config else 1

        return (max_layer_params * bytes_per_param) // tp_size

    def _estimate_num_params(self, hf_config: Any) -> int:
        """Rough estimate of total parameter count from HF config."""
        hidden_size = getattr(hf_config, "hidden_size", 0)
        num_layers = getattr(hf_config, "num_hidden_layers", 0)
        vocab_size = getattr(hf_config, "vocab_size", 0)
        intermediate_size = getattr(hf_config, "intermediate_size", 0)

        if hidden_size == 0 or num_layers == 0:
            return 0

        total = 0

        # Embedding
        total += vocab_size * hidden_size

        # Per-layer attention
        kv_lora_rank = getattr(hf_config, "kv_lora_rank", 0)
        q_lora_rank = getattr(hf_config, "q_lora_rank", 0)
        if kv_lora_rank > 0 and q_lora_rank > 0:
            qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", 0)
            qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", 0)
            if qk_nope_head_dim == 0:
                qk_nope_head_dim = max(0, getattr(hf_config, "qk_head_dim", 0) - qk_rope_head_dim)
            v_head_dim = getattr(hf_config, "v_head_dim", qk_nope_head_dim)
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            num_heads = getattr(hf_config, "num_attention_heads", 1)
            total += num_layers * (
                hidden_size * q_lora_rank +
                q_lora_rank * (num_heads * qk_head_dim) +
                hidden_size * (kv_lora_rank + qk_rope_head_dim) +
                kv_lora_rank * (num_heads * (qk_nope_head_dim + v_head_dim)) +
                (num_heads * v_head_dim) * hidden_size +
                q_lora_rank + kv_lora_rank
            )
        else:
            num_heads = getattr(hf_config, "num_attention_heads", 1)
            head_dim = getattr(hf_config, "head_dim", 0) or hidden_size // num_heads
            # Attention weights: q_proj + k_proj + v_proj + o_proj
            # q_proj: [hidden, num_heads * head_dim]
            # k_proj: [hidden, num_kv_heads * head_dim]
            # v_proj: [hidden, num_kv_heads * head_dim]
            # o_proj: [num_heads * head_dim, hidden]
            # Simplified: ~4 * hidden * (num_heads * head_dim)
            total += num_layers * 4 * hidden_size * (num_heads * head_dim)

        # Per-layer MLP
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)
        n_routed = getattr(hf_config, "n_routed_experts", None) or getattr(hf_config, "num_experts", 0)
        moe_inter = getattr(hf_config, "moe_intermediate_size", intermediate_size)
        n_shared = getattr(hf_config, "n_shared_experts", 0)

        dense_layers = min(first_k_dense, num_layers)
        total += dense_layers * 3 * hidden_size * intermediate_size

        moe_layers = num_layers - dense_layers
        if n_routed > 0:
            total += moe_layers * n_routed * 3 * hidden_size * moe_inter
        else:
            total += moe_layers * 3 * hidden_size * intermediate_size
        if n_shared > 0:
            total += moe_layers * n_shared * 3 * hidden_size * moe_inter

        tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
        if not tie_word_embeddings:
            total += vocab_size * hidden_size

        total += hidden_size * 2

        return total

    def _get_bytes_per_param(self, model_config: Any, vllm_config: Any) -> int:
        """Determine bytes per parameter based on dtype and quantization.

        For vllm-ascend quantized models, the quantization string is stored in
        model_config.quantization (e.g., "ascend" for W8A8). We also check
        vllm_config.quant_config and HF config for quantization hints.
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
            if "gptq" in quant_method or "awq" in quant_method:
                return 1

        # Fallback to model dtype
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
