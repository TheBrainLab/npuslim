"""NPUSlim vLLM model entry for Kimi-K2 MCore v2.

This adapter keeps the vLLM runtime shell (PP/TP collectives, KV cache, model
wrapper) but owns the model-side semantics whenever they diverge from
vllm.model_executor.models.deepseek_v2. In particular, the MLP / MoE path
follows the fixed modeling implementation, including grouped-GEMM dispatch for
expert execution when the NPU kernel is available.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import ignore_torch_compile
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.utils import is_pp_missing_parameter, sequence_parallel_chunk

logger = init_logger(__name__)

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    from megatron.core.fusions.fused_bias_swiglu import (
        bias_swiglu_impl as megatron_bias_swiglu_impl,
    )
except ImportError:
    megatron_bias_swiglu_impl = None

try:
    from mindspeed.ops.npu_rotary_position_embedding import (
        npu_rotary_position_embedding as mindspeed_npu_rotary_position_embedding,
    )
except ImportError:
    mindspeed_npu_rotary_position_embedding = None

try:
    from mindspeed.core.fusions.grouped_matmul import (
        Ops as mindspeed_grouped_matmul_ops,
    )
except ImportError:
    mindspeed_grouped_matmul_ops = None

_OPTIONAL_MISSING_SUFFIXES = (
    ".self_attn.q_layernorm.bias",
    ".self_attn.k_layernorm.bias",
    ".mlp.gate.bias",
    ".mlp.gate.expert_bias",
)


class _RuntimeOps:
    def __init__(self) -> None:
        self.npu_rms_norm = (
            getattr(torch_npu, "npu_rms_norm", None) if torch_npu is not None else None
        )
        self.npu_rotary_position_embedding = mindspeed_npu_rotary_position_embedding
        self.grouped_gemm = (
            getattr(mindspeed_grouped_matmul_ops, "gmm", None)
            if mindspeed_grouped_matmul_ops is not None
            else None
        )
        self.grouped_gemm_signature = (
            inspect.signature(self.grouped_gemm)
            if callable(self.grouped_gemm)
            else None
        )

    def grouped_gemm_kwargs(
        self, original_weight: torch.Tensor, gemm_fusion: bool
    ) -> dict[str, Any]:
        if self.grouped_gemm_signature is None:
            return {}
        kwargs: dict[str, Any] = {}
        if "trans_b" in self.grouped_gemm_signature.parameters:
            kwargs["trans_b"] = False
        if "gemm_fusion" in self.grouped_gemm_signature.parameters:
            kwargs["gemm_fusion"] = gemm_fusion
        if "original_weight" in self.grouped_gemm_signature.parameters:
            kwargs["original_weight"] = original_weight
        return kwargs


RUNTIME = _RuntimeOps()


def bias_swiglu_impl(
    input_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    fp8_input_store: bool = False,
) -> torch.Tensor:
    del fp8_input_store
    if megatron_bias_swiglu_impl is not None:
        return megatron_bias_swiglu_impl(input_tensor, bias)

    flat_input = input_tensor.view(-1, input_tensor.shape[-1])
    if bias is not None:
        flat_input = flat_input + bias
    gate, up = torch.chunk(flat_input, 2, dim=-1)
    output = (F.silu(gate.float()) * up.float()).to(dtype=input_tensor.dtype)
    if input_tensor.dim() == 2:
        return output
    return output.view(*input_tensor.shape[:-1], output.shape[-1])


def npu_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, object]:
    if RUNTIME.npu_rms_norm is not None:
        return RUNTIME.npu_rms_norm(hidden_states, weight, epsilon=epsilon)

    hidden_fp32 = hidden_states.float()
    weight_fp32 = weight.float()
    normed_fp32 = hidden_fp32 * torch.rsqrt(
        hidden_fp32.pow(2).mean(dim=-1, keepdim=True) + epsilon
    )
    return (normed_fp32 * weight_fp32).to(hidden_states.dtype), None


def _run_grouped_gemm_single_expert(
    inputs: torch.Tensor,
    weight_2d: torch.Tensor,
    gemm_fusion: bool = False,
) -> torch.Tensor:
    if RUNTIME.grouped_gemm is None:
        return torch.matmul(inputs, weight_2d)

    tokens_per_expert = torch.tensor(
        [inputs.shape[0]], device=inputs.device, dtype=torch.int64
    )
    weights = weight_2d.unsqueeze(0).contiguous()
    original_weight = weight_2d.contiguous()
    kwargs = RUNTIME.grouped_gemm_kwargs(original_weight, gemm_fusion)
    return RUNTIME.grouped_gemm(inputs, weights, tokens_per_expert, **kwargs)


def npu_rotary_position_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mode: int = 0,
) -> torch.Tensor:
    if RUNTIME.npu_rotary_position_embedding is not None:
        return RUNTIME.npu_rotary_position_embedding(x, cos, sin, mode)
    if mode == 0:
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x.float()[..., ::2]
        x2 = x.float()[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return (x.float() * cos.float() + rotated * sin.float()).to(dtype=x.dtype)


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low_dim = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high_dim = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    low = max(min(low_dim, high_dim), 0)
    high = min(max(low_dim, high_dim), dim - 1)
    return low, high


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(
    min_value: int,
    max_value: int,
    dim: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if min_value == max_value:
        max_value += 0.001
    linear_func = (
        torch.arange(dim, dtype=torch.float32, device=device) - min_value
    ) / (max_value - min_value)
    return torch.clamp(linear_func, 0, 1)


def _build_megatron_rope_inv_freq(
    config: Any,
    head_dim: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    base = float(getattr(config, "rope_theta", 10000.0))
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freq_extra = (1.0 / (base ** (idx / head_dim))).contiguous()

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None or rope_scaling.get("type") != "yarn":
        return freq_extra

    scaling_factor = float(rope_scaling["factor"])
    beta_fast = float(rope_scaling.get("beta_fast", 32))
    beta_slow = float(rope_scaling.get("beta_slow", 1))
    original_max_pos = int(rope_scaling.get("original_max_position_embeddings", 4096))
    freq_inter = (1.0 / (scaling_factor * (base ** (idx / head_dim)))).contiguous()
    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        head_dim,
        base,
        original_max_pos,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(
        low,
        high,
        head_dim // 2,
        device=device,
    )
    return (freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask).contiguous()


def _build_megatron_rope_mscale(config: Any) -> float:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None or rope_scaling.get("type") != "yarn":
        return 1.0
    return float(
        yarn_get_mscale(
            float(rope_scaling["factor"]),
            float(rope_scaling.get("mscale", 1.0)),
        )
        / yarn_get_mscale(
            float(rope_scaling["factor"]),
            float(rope_scaling.get("mscale_all_dim", 0.0)),
        )
    )


def _apply_megatron_rope_flat(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    inv_freq: torch.Tensor,
    mscale: float,
    rotary_interleaved: bool,
) -> torch.Tensor:
    num_tokens = tensor.shape[0]
    tensor_3d = tensor.view(num_tokens, num_heads, head_dim)
    inv_freq = inv_freq.to(tensor.device)
    freqs = torch.outer(positions.to(tensor.device, dtype=torch.float32), inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    cos = (torch.cos(freqs) * mscale).to(tensor_3d.dtype)
    sin = (torch.sin(freqs) * mscale).to(tensor_3d.dtype)
    mode = 1 if rotary_interleaved else 0
    rotated = npu_rotary_position_embedding(tensor_3d.contiguous(), cos, sin, mode)
    return rotated.view(num_tokens, num_heads * head_dim)


class KimiK2MCoreV2MegatronRoPE(nn.Module):
    def __init__(self, config: Any, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mscale = _build_megatron_rope_mscale(config)
        self.rotary_interleaved = bool(getattr(config, "rotary_interleaved", False))
        inv_freq = _build_megatron_rope_inv_freq(config, head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.reshape(-1).to(dtype=torch.long)
        q = _apply_megatron_rope_flat(
            q,
            positions,
            num_heads=num_heads,
            head_dim=self.head_dim,
            inv_freq=self.inv_freq,
            mscale=self.mscale,
            rotary_interleaved=self.rotary_interleaved,
        )
        k = _apply_megatron_rope_flat(
            k,
            positions,
            num_heads=num_kv_heads,
            head_dim=self.head_dim,
            inv_freq=self.inv_freq,
            mscale=self.mscale,
            rotary_interleaved=self.rotary_interleaved,
        )
        return q, k


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return output + self.bias


class KimiK2MCoreV2RMSNorm(nn.Module):
    """Baseline-compatible RMSNorm with vLLM residual-call support."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states

        weight = self.weight.to(hidden_states.dtype)
        output, _ = npu_rms_norm(hidden_states, weight, self.variance_epsilon)
        if residual is None:
            return output
        return output, residual


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

    if not hasattr(hf_config, "rope_parameters") or not getattr(
        hf_config, "rope_parameters"
    ):
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
            "Unexpected fused_qkv rows for Kimi-K2-MCore V2: "
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


def _row_parallel_reduce(layer: RowParallelLinear, output: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "reduce_results", True) and getattr(layer, "tp_size", 1) > 1:
        return tensor_model_parallel_all_reduce(output)
    return output


class KimiK2MCoreV2MLP(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        quant_config: Any = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul() if SiluAndMul is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_prob(x, None)

    def forward_with_prob(
        self,
        x: torch.Tensor,
        prob: torch.Tensor | None,
    ) -> torch.Tensor:
        use_grouped_gemm = prob is not None and hasattr(self.gate_up_proj, "weight")
        gate_up_weight = getattr(self.gate_up_proj, "weight", None)
        down_weight = getattr(self.down_proj, "weight", None)

        if use_grouped_gemm and gate_up_weight is not None:
            gate_up = _run_grouped_gemm_single_expert(
                x,
                gate_up_weight.t().contiguous(),
                gemm_fusion=getattr(
                    self.config, "gemm_gradient_accumulation_fusion", False
                ),
            )
        elif gate_up_weight is not None:
            gate_up = F.linear(x, gate_up_weight, bias=None)
        else:
            gate_up, _ = self.gate_up_proj(x)

        if self.config.hidden_act == "silu":
            intermediate = bias_swiglu_impl(gate_up, None)
        else:
            gate, up = torch.chunk(gate_up, 2, dim=-1)
            intermediate = getattr(F, self.config.hidden_act)(gate) * up

        if prob is not None:
            intermediate = intermediate * prob.unsqueeze(-1).to(intermediate.dtype)

        if prob is not None and down_weight is not None:
            output = _run_grouped_gemm_single_expert(
                intermediate,
                down_weight.t().contiguous(),
                gemm_fusion=getattr(
                    self.config, "gemm_gradient_accumulation_fusion", False
                ),
            )
            return _row_parallel_reduce(self.down_proj, output)

        if down_weight is not None:
            output = F.linear(intermediate, down_weight, bias=None)
            return _row_parallel_reduce(self.down_proj, output)

        output, _ = self.down_proj(intermediate)
        return output


class KimiK2MCoreV2MoEGate(nn.Module):
    def __init__(self, config: Any, prefix: str = ""):
        super().__init__()
        del prefix
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = getattr(config, "seq_aux", False)
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.moe_router_enable_expert_bias = getattr(
            config, "moe_router_enable_expert_bias", False
        )
        self.moe_router_dtype = getattr(config, "moe_router_dtype", "fp32")
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.expert_bias = (
            nn.Parameter(torch.empty((self.n_routed_experts)))
            if self.moe_router_enable_expert_bias
            else None
        )
        self.bias = (
            nn.Parameter(torch.empty((self.n_routed_experts)))
            if self.moe_router_enable_expert_bias
            else None
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.expert_bias is not None:
            init.zeros_(self.expert_bias)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _resolve_router_bias(self) -> torch.Tensor | None:
        expert_bias = None
        legacy_bias = None
        if self.expert_bias is not None and torch.isfinite(self.expert_bias).all():
            expert_bias = self.expert_bias
        if self.bias is not None and torch.isfinite(self.bias).all():
            legacy_bias = self.bias
        if expert_bias is not None:
            if legacy_bias is None:
                return expert_bias
            if torch.count_nonzero(expert_bias).item() != 0:
                return expert_bias
            if torch.count_nonzero(legacy_bias).item() != 0:
                return legacy_bias
            return expert_bias
        return legacy_bias

    def _get_capacity(self, num_tokens: int, num_experts: int, capacity_factor: float) -> int:
        return math.ceil((num_tokens / num_experts) * capacity_factor)

    def _topk_with_capacity(
        self,
        scores: torch.Tensor,
        router_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens, num_experts = scores.shape
        scores_for_routing = scores
        if router_bias is not None:
            scores_for_routing = scores_for_routing + router_bias.to(scores.dtype).unsqueeze(0)

        if getattr(self.config, "moe_router_group_topk", 0):
            per_group_topk = max(1, self.top_k // self.config.moe_router_group_topk)
            group_scores = (
                scores_for_routing.view(num_tokens, self.config.moe_router_num_groups, -1)
                .topk(per_group_topk, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.config.moe_router_group_topk, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    num_tokens,
                    self.config.moe_router_num_groups,
                    num_experts // self.config.moe_router_num_groups,
                )
                .reshape(num_tokens, -1)
            )
            masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
            _, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        else:
            _, top_indices = torch.topk(scores_for_routing, k=self.top_k, dim=-1)

        selected_scores = torch.gather(scores, dim=1, index=top_indices).type_as(scores)
        probs = (
            selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
            if self.top_k > 1
            else selected_scores
        )
        scaling_factor = getattr(self.config, "moe_router_topk_scaling_factor", None)
        if scaling_factor:
            probs = probs * scaling_factor

        topk_masked_gates = torch.zeros_like(scores).scatter(1, top_indices, probs)
        topk_map = torch.zeros_like(scores).int().scatter(1, top_indices, 1).bool()
        tokens_per_expert = topk_map.sum(dim=0)

        capacity_factor = getattr(self.config, "moe_expert_capacity_factor", None)
        pad_to_capacity = getattr(self.config, "moe_pad_expert_input_to_capacity", False)
        drop_policy = getattr(self.config, "moe_token_drop_policy", "probs")
        if capacity_factor is None:
            return topk_masked_gates, topk_map, tokens_per_expert

        expert_capacity = self._get_capacity(
            num_tokens=num_tokens * self.top_k,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(scores).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(
                topk_map.int(), k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(scores).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.moe_router_dtype == "bf16":
            router_dtype = torch.bfloat16
        elif self.moe_router_dtype == "fp16":
            router_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported moe_router_dtype: {self.moe_router_dtype}")

        logits = F.linear(
            hidden_states.to(router_dtype),
            self.weight.to(router_dtype),
            None,
        ).to(torch.float32)

        if self.scoring_func != "sigmoid":
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        scores = logits.sigmoid()

        router_bias = self._resolve_router_bias()
        final_probs, _, _ = self._topk_with_capacity(scores, router_bias)
        topk_idx = torch.topk(final_probs, k=self.top_k, dim=-1).indices
        topk_weight = torch.gather(final_probs, dim=1, index=topk_idx)
        return topk_idx, topk_weight, None


class KimiK2MCoreV2MoE(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        parallel_config: Any,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_sequence_parallel = getattr(parallel_config, "use_sequence_parallel_moe", False)

        ep_info = get_ep_group() if callable(get_ep_group) else None
        if ep_info is not None:
            self.ep_group = ep_info.device_group
            self.ep_rank = ep_info.rank_in_group
            self.ep_size = self.ep_group.size()
        else:
            self.ep_group = None
            self.ep_rank = 0
            self.ep_size = 1

        if self.config.n_routed_experts % self.ep_size != 0:
            raise ValueError(
                f"n_routed_experts={self.config.n_routed_experts} must be divisible by ep_size={self.ep_size}"
            )
        self.experts_per_rank = self.config.n_routed_experts // self.ep_size
        local_start = self.ep_rank * self.experts_per_rank
        local_end = local_start + self.experts_per_rank
        self.experts = nn.ModuleDict(
            {
                str(i): KimiK2MCoreV2MLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                    quant_config=quant_config,
                    reduce_results=True,
                    is_sequence_parallel=self.is_sequence_parallel,
                    prefix=f"{prefix}.experts.{i}",
                )
                for i in range(local_start, local_end)
            }
        )
        self.gate = KimiK2MCoreV2MoEGate(config, prefix=f"{prefix}.gate")
        if config.n_shared_experts is not None:
            self.shared_experts = KimiK2MCoreV2MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                quant_config=quant_config,
                reduce_results=True,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        else:
            flat_hidden_states = hidden_states

        topk_idx, topk_weight, _ = self.gate(hidden_states)
        if self.is_sequence_parallel:
            flat_hidden_states = sequence_parallel_chunk(flat_hidden_states)

        if self.ep_size == 1:
            output = self.moe_forward(flat_hidden_states, topk_idx, topk_weight)
        else:
            output = self.moe_infer(flat_hidden_states, topk_idx, topk_weight)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity.view(-1, identity.shape[-1]))
            output = output + shared_output

        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[: flat_hidden_states.shape[0]]

        if len(orig_shape) == 3:
            return output.view(*orig_shape)
        return output

    def moe_forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        out = x.new_zeros(x.shape)
        for expert_id in range(self.config.n_routed_experts):
            expert_key = str(expert_id)
            if expert_key not in self.experts:
                continue
            expert = self.experts[expert_key]
            mask = topk_ids == expert_id
            token_idx, which = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_in = x[token_idx]
            expert_w = topk_weight[token_idx, which].to(expert_in.dtype)
            expert_out = expert.forward_with_prob(expert_in, expert_w)
            out[token_idx] = out[token_idx] + expert_out
        return out

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.config.n_routed_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        if self.ep_size > 1:
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.ep_group)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(),
                sorted_tokens.shape[1],
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
                group=self.ep_group,
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size,
                self.experts_per_rank,
            ).sum(dim=0)
            gathered_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gathered_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gathered_idxs = gathered_idxs.argsort()
            sorted_tokens = gathered_tokens[gathered_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        else:
            gathered_idxs = None
            input_split_sizes = None
            output_splits = None

        tokens_per_expert_np = tokens_per_expert.cpu().numpy()
        flat_probs = topk_weight.view(-1)[idxs]
        outputs = []
        start_idx = 0
        for local_idx, num_tokens in enumerate(tokens_per_expert_np):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            global_expert_id = local_idx + self.ep_rank * self.experts_per_rank
            expert = self.experts[str(global_expert_id)]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            token_probs = flat_probs[start_idx:end_idx].to(tokens_for_this_expert.dtype)
            expert_out = expert.forward_with_prob(tokens_for_this_expert, token_probs)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            reordered = torch.empty_like(outs)
            reordered[gathered_idxs] = outs
            gathered_tokens = reordered.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(reordered.split(output_splits)),
                group=self.ep_group,
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class KimiK2MCoreV2Attention(nn.Module):
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
            getattr(config, "num_query_groups", getattr(config, "num_key_value_heads", num_heads))
        )
        self.head_dim = int(getattr(config, "kv_channels", hidden_size // self.total_num_heads))
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
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
        del max_position_embeddings
        self.rotary_emb = KimiK2MCoreV2MegatronRoPE(config, self.head_dim)
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

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_layernorm is not None:
            q = self.q_layernorm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        if self.k_layernorm is not None:
            k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class KimiK2MCoreV2DecoderLayer(deepseek_v2.DeepseekV2DecoderLayer):
    def __init__(
        self,
        vllm_config: Any,
        prefix: str,
        config: Any | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        del topk_indices_buffer
        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.use_mha = True
        self.self_attn = KimiK2MCoreV2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (
            config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % moe_layer_freq == 0
        ):
            self.mlp = KimiK2MCoreV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = KimiK2MCoreV2MLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = KimiK2MCoreV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = KimiK2MCoreV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
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
        if isinstance(self.mlp, KimiK2MCoreV2MLP) and hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states, residual


# @ignore_torch_compile
@deepseek_v2.support_torch_compile
class KimiK2MCoreV2Model(deepseek_v2.DeepseekV2Model):
    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = deepseek_v2.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        topk_indices_buffer = None
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        pp_group = get_pp_group()
        if pp_group is not None and pp_group.is_first_rank:
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
            lambda prefix: KimiK2MCoreV2DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if pp_group is not None and pp_group.is_last_rank:
            self.norm = KimiK2MCoreV2RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = deepseek_v2.PPMissingLayer()
        self.make_empty_intermediate_tensors = deepseek_v2.make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers = ()


class KimiK2MCoreV2ForCausalLM(deepseek_v2.DeepseekV3ForCausalLM):
    model_cls = KimiK2MCoreV2Model

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        _prepare_kimi_k2_mcore_hf_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fused_params_mapping = [
            ("qkv_proj", "fused_qkv"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight, shard_id)
                except TypeError:
                    weight_loader(param, loaded_weight)
                loaded_params.add(name_mapped)
                handled = True
                break
            if handled:
                continue

            handled = False
            for param_name, weight_name in fused_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
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
                loaded_params.add(name_mapped)
                handled = True
                break
            if handled:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None or is_pp_missing_parameter(name, self) or name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        optional_missing = {
            name
            for name in params_dict
            if name.endswith(_OPTIONAL_MISSING_SUFFIXES) and name not in loaded_params
        }
        if optional_missing:
            with torch.no_grad():
                for name in optional_missing:
                    params_dict[name].zero_()
            loaded_params.update(optional_missing)
            logger.warning(
                "Initialized %d missing optional tensors to zeros.",
                len(optional_missing),
            )
        return loaded_params
