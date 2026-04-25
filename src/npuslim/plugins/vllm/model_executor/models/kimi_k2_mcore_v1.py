"""NPUSlim vLLM model entry for Kimi-K2 MCore (v1 / pre-router-patch version).

This is the previous iteration kept for A/B comparison. Differences from the
current version (kimi_k2_mcore.py):
- No MoE expert-capacity / token-drop router patching.
- Uses custom _QKLayerNormNoBias instead of vLLM RMSNorm for qk norm.
- Simpler HF config normalisation (no capacity-factor / token-drop defaults).
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models import deepseek_v2

logger = init_logger(__name__)

_OPTIONAL_MISSING_BIAS_SUFFIXES = (
    ".self_attn.q_layernorm.bias",
    ".self_attn.k_layernorm.bias",
    ".mlp.gate.bias",
)


class _QKLayerNormNoBias(nn.Module):
    """LayerNorm without bias for q/k layernorm compatibility."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = hidden_states * self.weight
        return hidden_states.to(input_dtype)


def _build_rope_parameters_from_hf_config(hf_config: Any) -> dict[str, Any]:
    """Build vLLM rope_parameters from HF config fields."""
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rope_params: dict[str, Any] = {}

    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        rope_params["rope_theta"] = rope_theta

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if rope_type is not None:
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


def _build_qk_norm_layer(head_dim: int, eps: float, hf_config: Any) -> nn.Module:
    use_rmsnorm = bool(getattr(hf_config, "use_rmsnorm_for_qk", False))
    if use_rmsnorm:
        return deepseek_v2.RMSNorm(head_dim, eps=eps)
    return _QKLayerNormNoBias(head_dim, eps=eps)


def _prepare_kimi_k2_mcore_hf_config(hf_config: Any) -> None:
    """Normalize HF config to Kimi-K2-MCore GQA behavior."""
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


class KimiK2MCoreV1Attention(nn.Module):
    """GQA attention for Kimi-K2-MCore with optional q/k per-head norm."""

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
            self.q_layernorm = _build_qk_norm_layer(self.head_dim, eps, config)
            self.k_layernorm = _build_qk_norm_layer(self.head_dim, eps, config)
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


class KimiK2MCoreV1DecoderLayer(deepseek_v2.DeepseekV2DecoderLayer):
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
        self.self_attn = KimiK2MCoreV1Attention(
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
class KimiK2MCoreV1Model(deepseek_v2.DeepseekV2Model):
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
            lambda prefix: KimiK2MCoreV1DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
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


class KimiK2MCoreV1ForCausalLM(deepseek_v2.DeepseekV3ForCausalLM):
    """Runtime model for `architectures = ["KimiK2MCoreV1ForCausalLM"]`."""

    model_cls = KimiK2MCoreV1Model

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        _prepare_kimi_k2_mcore_hf_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, object]]) -> set[str]:
        loaded = super().load_weights(weights)

        # Some converted checkpoints do not store these bias tensors.
        params_dict = dict(self.named_parameters())
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
