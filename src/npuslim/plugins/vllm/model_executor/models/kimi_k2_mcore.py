"""NPUSlim vLLM model entry for Kimi-K2 MCore.

This adapter keeps the stable vLLM DeepSeek-V2/3 execution structure used by
the previous Kimi plugin, while aligning adapter-owned runtime behavior with
the fixed reference modeling:
- grouped-query attention with q/k RMSNorm
- RoPE parameters normalized from the HF config
- runtime helpers prefer NPU ops when they are available
- missing optional bias tensors are zero-initialized during weight loading
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.utils import is_pp_missing_parameter

logger = init_logger(__name__)

try:
    import torch_npu
except ImportError:
    torch_npu = None

_OPTIONAL_MISSING_BIAS_SUFFIXES = (
    ".self_attn.q_layernorm.bias",
    ".self_attn.k_layernorm.bias",
    ".mlp.gate.bias",
)

_NPU_RMS_NORM = getattr(torch_npu, "npu_rms_norm", None) if torch_npu is not None else None


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _NPU_RMS_NORM is not None:
            output, _ = _NPU_RMS_NORM(
                hidden_states, self.weight, epsilon=self.variance_epsilon,
            )
            return output + self.bias

        hidden_fp32 = hidden_states.float()
        weight_fp32 = self.weight.float()
        normed_fp32 = hidden_fp32 * torch.rsqrt(
            hidden_fp32.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (normed_fp32 * weight_fp32).to(hidden_states.dtype) + self.bias


def _build_rope_parameters_from_hf_config(hf_config: Any) -> dict[str, Any]:
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rope_params: dict[str, Any] = {}

    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        rope_params["rope_theta"] = rope_theta

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if rope_type is not None:
        # Kimi-K2 MCore checkpoints store HF-style "yarn", but the reference
        # train/baseline path matches vLLM's DeepSeek-specific rotary variant.
        if rope_type == "yarn":
            rope_type = "deepseek_yarn"
        rope_params["rope_type"] = rope_type

    for key in (
        "factor",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "original_max_position_embeddings",
        "short_factor",
        "long_factor",
        "low_freq_factor",
        "high_freq_factor",
    ):
        if key in rope_scaling:
            rope_params[key] = rope_scaling[key]

    return rope_params


def _prepare_kimi_k2_mcore_hf_config(hf_config: Any) -> None:
    """Normalize HF config to the Kimi-K2 MCore vLLM execution path."""
    num_query_groups = getattr(hf_config, "num_query_groups", None)
    if num_query_groups is not None:
        num_query_groups = int(num_query_groups)
        if getattr(hf_config, "num_key_value_heads", None) != num_query_groups:
            hf_config.num_key_value_heads = num_query_groups
            logger.info(
                "Set num_key_value_heads=%d from num_query_groups for Kimi-K2-MCore.",
                num_query_groups,
            )

    for attr, value in (
        ("q_lora_rank", None),
        ("kv_lora_rank", 0),
        ("qk_nope_head_dim", 0),
        ("qk_rope_head_dim", 0),
        ("v_head_dim", 0),
    ):
        if getattr(hf_config, attr, None) != value:
            setattr(hf_config, attr, value)

    rope_params = _build_rope_parameters_from_hf_config(hf_config)
    if rope_params:
        hf_config.rope_parameters = rope_params


def _reorder_fused_qkv_weight_for_vllm(
    fused_qkv: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert Megatron GQA-interleaved fused_qkv rows into vLLM Q|K|V order."""
    # Checkpoint layout follows the fused Megatron/GQA grouping:
    # [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]
    # vLLM QKVParallelLinear expects contiguous blocks instead:
    # [all Q | all K | all V]
    heads_per_group = num_attention_heads // num_query_groups
    rows_per_group = (heads_per_group + 2) * head_dim
    expected_rows = num_query_groups * rows_per_group
    if fused_qkv.shape[0] != expected_rows:
        raise ValueError(
            "Unexpected fused_qkv rows for Kimi-K2-MCore: "
            f"got {fused_qkv.shape[0]}, expected {expected_rows} "
            f"(num_attention_heads={num_attention_heads}, "
            f"num_query_groups={num_query_groups}, head_dim={head_dim})"
        )

    grouped = fused_qkv.view(num_query_groups, rows_per_group, *fused_qkv.shape[1:])
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_attention_heads * head_dim, *fused_qkv.shape[1:]),
            k.reshape(num_query_groups * head_dim, *fused_qkv.shape[1:]),
            v.reshape(num_query_groups * head_dim, *fused_qkv.shape[1:]),
        ],
        dim=0,
    )


class KimiK2MCoreAttention(nn.Module):
    """Grouped-query attention with q/k RMSNorm for Kimi-K2 MCore."""

    def __init__(
        self,
        *,
        config: Any,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 8192,
        cache_config: Any = None,
        quant_config: Any = None,
        prefix: str = "",
        **_: Any,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = int(
            getattr(
                config,
                "num_query_groups",
                getattr(config, "num_key_value_heads", num_heads),
            )
        )
        self.head_dim = int(
            getattr(config, "kv_channels", hidden_size // self.total_num_heads)
        )

        tp_size = deepseek_v2.get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = deepseek_v2.QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = deepseek_v2.RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        if not rope_parameters:
            rope_parameters = _build_rope_parameters_from_hf_config(config)
        self.rotary_emb = deepseek_v2.get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.attn = deepseek_v2.Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        if bool(getattr(config, "qk_layernorm", False)):
            self.q_layernorm = DeepseekV3RMSNorm(self.head_dim, eps=eps)
            self.k_layernorm = DeepseekV3RMSNorm(self.head_dim, eps=eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.q_layernorm is not None:
            q = q.view(-1, self.num_heads, self.head_dim)
            q = self.q_layernorm(q)
            q = q.view(-1, self.q_size)

        if self.k_layernorm is not None:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            k = self.k_layernorm(k)
            k = k.view(-1, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

        return output


def _baseline_grouped_topk_with_capacity(
    router_logits: torch.Tensor,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    capacity_factor: float | None,
    pad_to_capacity: bool,
    drop_policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing with capacity-based token dropping.

    Ported from ``KimiK2MCoreV2MoEGate._topk_with_capacity`` to align V1
    routing with the baseline.  Returns ``(topk_weights, topk_ids)`` *without*
    ``routed_scaling_factor`` — ``DeepseekV2MoE.forward`` applies it post-hoc.
    """
    num_tokens, num_experts = router_logits.shape
    scores = router_logits.float().sigmoid()

    # Grouped expert selection (no bias — baseline doesn't use it)
    if num_expert_group > 0 and topk_group > 0:
        per_group_topk = max(1, top_k // topk_group)
        group_scores = (
            scores.view(num_tokens, num_expert_group, -1)
            .topk(per_group_topk, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, num_expert_group, num_experts // num_expert_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        _, top_indices = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    else:
        _, top_indices = torch.topk(scores, k=top_k, dim=-1)

    # Normalize
    selected_scores = torch.gather(scores, dim=1, index=top_indices)
    probs = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)

    if capacity_factor is None:
        return probs.to(torch.float32), top_indices.to(torch.int32)

    # Capacity-based token dropping
    topk_masked_gates = torch.zeros_like(scores).scatter(1, top_indices, probs)
    topk_map = (
        torch.zeros_like(scores).int().scatter(1, top_indices, 1).bool()
    )

    expert_capacity = min(
        math.ceil((num_tokens * top_k / num_experts) * capacity_factor),
        num_tokens,
    )

    if drop_policy == "probs":
        _, capacity_indices = torch.topk(
            topk_masked_gates, k=expert_capacity, dim=0, sorted=False
        )
    else:
        _, capacity_indices = torch.topk(
            topk_map.int(), k=expert_capacity, dim=0, sorted=False
        )
    capacity_mask = (
        torch.zeros_like(scores).scatter(0, capacity_indices, 1).bool()
    )

    final_map = capacity_mask if pad_to_capacity else torch.logical_and(topk_map, capacity_mask)
    final_probs = topk_masked_gates * final_map

    topk_idx = torch.topk(final_probs, k=top_k, dim=-1).indices
    topk_weight = torch.gather(final_probs, dim=1, index=topk_idx)
    return topk_weight.to(torch.float32), topk_idx.to(torch.int32)


class KimiK2MCoreDecoderLayer(deepseek_v2.DeepseekV2DecoderLayer):
    """Decoder layer overriding attention with Kimi-K2-MCore GQA attention."""

    def __init__(
        self,
        vllm_config: Any,
        prefix: str,
        config: Any | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.use_mha = True
        self.self_attn = KimiK2MCoreAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = deepseek_v2.DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

            # Replace router with baseline-style routing that includes
            # capacity-based token dropping (the key difference between
            # vLLM's grouped_topk and the baseline).
            _router = self.mlp.experts.router
            _top_k = _router.top_k
            _num_groups = getattr(config, "moe_router_num_groups", 0)
            _topk_group = getattr(config, "moe_router_group_topk", 0)
            _cap_factor = getattr(config, "moe_expert_capacity_factor", None)
            _pad_cap = getattr(config, "moe_pad_expert_input_to_capacity", False)
            _drop_policy = getattr(config, "moe_token_drop_policy", "probs")

            def _baseline_routing(
                _self, hidden_states, router_logits, indices_type,
                __top_k=_top_k, __ng=_num_groups, __tg=_topk_group,
                __cf=_cap_factor, __pc=_pad_cap, __dp=_drop_policy,
            ):
                return _baseline_grouped_topk_with_capacity(
                    router_logits=router_logits,
                    top_k=__top_k,
                    num_expert_group=__ng,
                    topk_group=__tg,
                    capacity_factor=__cf,
                    pad_to_capacity=__pc,
                    drop_policy=__dp,
                )

            import types as _types
            _router._compute_routing = _types.MethodType(
                _baseline_routing, _router,
            )
        else:
            self.mlp = deepseek_v2.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = deepseek_v2.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = deepseek_v2.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del llama_4_scaling

        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, deepseek_v2.DeepseekV2MLP) and (
            hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@deepseek_v2.support_torch_compile
class KimiK2MCoreModel(deepseek_v2.DeepseekV2Model):
    """DeepSeek model wrapper with Kimi-K2-MCore decoder layer."""

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = deepseek_v2.current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if deepseek_v2.get_pp_group().is_first_rank:
            self.embed_tokens = deepseek_v2.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = deepseek_v2.PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = deepseek_v2.make_layers(
            config.num_hidden_layers,
            lambda prefix: KimiK2MCoreDecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if deepseek_v2.get_pp_group().is_last_rank:
            self.norm = deepseek_v2.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = deepseek_v2.PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            deepseek_v2.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        self.aux_hidden_state_layers = ()


class KimiK2MCoreForCausalLM(deepseek_v2.DeepseekV3ForCausalLM):
    """Runtime model for `architectures = ["KimiK2MCoreForCausalLM"]`."""

    model_cls = KimiK2MCoreModel

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        _prepare_kimi_k2_mcore_hf_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        deferred_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if "fused_qkv" not in name:
                deferred_weights.append((name, loaded_weight))
                continue

            name_mapped = name.replace("fused_qkv", "qkv_proj")
            if is_pp_missing_parameter(name_mapped, self):
                continue
            if name_mapped not in params_dict:
                continue
            param = params_dict[name_mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            reordered_weight = _reorder_fused_qkv_weight_for_vllm(
                loaded_weight,
                num_attention_heads=int(self.config.num_attention_heads),
                num_query_groups=int(self.config.num_query_groups),
                head_dim=int(self.config.kv_channels),
            )

            weight_loader(param, reordered_weight)
            loaded.add(name_mapped)

        loaded.update(super().load_weights(deferred_weights))

        # Disable e_score_correction_bias in MoE routing to match the baseline's
        # grouped top-k logic. The baseline uses max-per-group scoring while
        # vLLM's noaux_tc path uses topk(2).sum() with bias correction. Kimi-K2
        # checkpoints carry e_score_correction_bias but the baseline model never
        # applies it during routing, so we disable it here to align the routing.
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "gate"):
                continue
            if hasattr(mlp.gate, "e_score_correction_bias"):
                mlp.gate.e_score_correction_bias = None
            if hasattr(mlp, "experts"):
                if hasattr(mlp.experts, "e_score_correction_bias"):
                    mlp.experts.e_score_correction_bias = None
                if hasattr(mlp.experts, "router") and hasattr(
                    mlp.experts.router, "e_score_correction_bias"
                ):
                    mlp.experts.router.e_score_correction_bias = None

        optional_missing = {
            name
            for name in params_dict
            if name.endswith(_OPTIONAL_MISSING_BIAS_SUFFIXES) and name not in loaded
        }
        if optional_missing:
            with torch.no_grad():
                for name in optional_missing:
                    params_dict[name].zero_()
            loaded.update(optional_missing)
            logger.warning(
                "Initialized %d missing optional bias tensors to zeros.",
                len(optional_missing),
            )

        return loaded
