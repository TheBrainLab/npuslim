"""Patch for vllm/model_executor/models/longcat_flash.py

Supports Grouped Routing (分组路由) from the custom HuggingFace model
``modeling_longcat_flash_group.py``.

Routing logic (matching HF ``LongcatFlashTopkRouter.get_topk_indices``):

    Router outputs (N+Z) logits → reshape into *F* groups of (N+Z)/F experts
    → pick best expert within each group → top-k from the group winners
    → map back to original expert indices in [0, N+Z).

The router classifier dimension stays at N+Z (no expansion).  Expert
computation uses N real + Z zero experts, unchanged.

Compatibility
-------------
- **Without zero experts** (``zero_expert_type=None``): sets
  ``custom_routing_function`` on the ``FusedMoE`` so the router factory
  creates a ``CustomRoutingRouter``.
- **With zero experts, GPU** (``zero_expert_type="identity"``): directly
  monkey-patches ``ZeroExpertRouter._compute_routing`` on the instance,
  because the router factory gives ``ZeroExpertRouter`` higher priority
  and it ignores ``custom_routing_function``.  Skipped on Ascend because
  ``AscendZeroExpertFusedMoE.forward()`` never calls this method.
- **Ascend NPU** (any ``zero_expert_type``): replaces ``select_experts``
  on the ``AscendZeroExpertFusedMoE`` instance, because its ``forward()``
  temporarily disables ``custom_routing_function`` and calls
  ``select_experts`` directly, bypassing both of the above paths.

Known Limitation
----------------
On Ascend NPU, the **second pass** of ``AscendZeroExpertFusedMoE.forward()``
(the ``AscendFusedMoE.forward`` call for real experts) uses Ascend's native
routing (standard top-k) rather than grouped routing, because the CANN fused
kernel does not support ``custom_routing_function``.  Only the first pass
(zero-expert selection via ``select_experts``) uses grouped routing.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import register_patch

target_logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Grouped routing function
# ---------------------------------------------------------------------------


def _grouped_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    expansion_factor: int,
    n_routed_experts: int,
    e_score_correction_bias: torch.Tensor | None = None,
    **_ignored,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing matching HF ``LongcatFlashTopkRouter``.

    Args:
        hidden_states: *(tokens, hidden_dim)* — unused, for API compatibility.
        gating_output: *(tokens, N+Z)* — raw router logits.
        topk: Number of experts to select.
        renormalize: Whether to normalize top-k weights to sum to 1.
        expansion_factor: *F*, number of experts per group.
        n_routed_experts: N+Z, total router output dimension.
        e_score_correction_bias: Optional bias added **after** softmax
            (matching HF behaviour).

    Returns:
        *topk_weights*: *(tokens, topk)*
        *topk_ids*: *(tokens, topk)* — expert indices in [0, N+Z).
    """
    del hidden_states  # unused — kept for API compatibility with FusedMoE
    total_groups = n_routed_experts // expansion_factor  # (N+Z) / F

    if total_groups < topk:
        raise ValueError(
            f"Grouped routing: expansion_factor={expansion_factor} yields "
            f"{total_groups} groups from {n_routed_experts} experts, "
            f"but topk={topk}. Reduce expansion_factor to at most "
            f"{n_routed_experts // topk}."
        )

    # 1. Softmax across all logits → scores
    scores = gating_output.softmax(dim=-1)

    # 2. Add bias after softmax (HF-style: scores + bias)
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    # 3. Reshape into F groups: (tokens, N+Z) → (tokens, F, total_groups)
    #    → transpose → (tokens, total_groups, F)
    grouped_scores = (
        scores_for_choice
        .view(-1, expansion_factor, total_groups)
        .transpose(-1, -2)
    )

    # 4. Best expert within each group
    #    group_score_best: (tokens, total_groups)
    #    group_best_idx:   (tokens, total_groups) — offset in [0, F)
    group_score_best, group_best_idx = grouped_scores.max(dim=-1)

    # 5. Top-k from the total_groups winners
    _, topk_group_ids = torch.topk(group_score_best, k=topk, dim=-1, sorted=False)

    # 6. Map back to original expert index:
    #    expert_idx = group_id + best_offset * total_groups
    best_offsets = group_best_idx.gather(1, topk_group_ids)
    topk_ids = topk_group_ids + best_offsets * total_groups

    # 7. Gather weights from the ORIGINAL softmax scores (before bias)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator

    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# ZeroExpertRouter patch helper
# ---------------------------------------------------------------------------


def _patch_zero_expert_router(experts, expansion_factor: int, n_routed: int):
    """Replace ``ZeroExpertRouter._compute_routing`` with grouped routing.

    The vLLM router factory prioritises ``ZeroExpertRouter`` over
    ``CustomRoutingRouter``, so ``custom_routing_function`` is ignored when
    ``zero_expert_type`` is set.  This helper directly replaces the routing
    logic while keeping the zero-expert bookkeeping intact.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        zero_experts_compute_triton,
    )

    router = experts.router  # ZeroExpertRouter instance

    def grouped_compute_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Grouped top-k (replaces fused_topk_bias call)
        topk_weights, topk_ids = _grouped_routing(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=router.top_k,
            renormalize=router.renormalize,
            expansion_factor=expansion_factor,
            n_routed_experts=n_routed,
            e_score_correction_bias=router.e_score_correction_bias,
        )
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(
            torch.int32 if indices_type is None else indices_type,
        )

        # 2. Scaling (same as original ZeroExpertRouter)
        if router.routed_scaling_factor != 1.0:
            topk_weights *= router.routed_scaling_factor

        # 3. Zero-expert contribution (same as original ZeroExpertRouter)
        router._zero_expert_output = zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=router.num_logical_experts,
            zero_expert_type=router.zero_expert_type,
            hidden_states=hidden_states,
        )

        # 4. Mask zero-expert IDs so downstream MoE ignores them
        zero_mask = topk_ids >= router.num_logical_experts
        topk_ids[zero_mask] = 0
        topk_weights[zero_mask] = 0.0

        return topk_weights, topk_ids

    router._compute_routing = grouped_compute_routing


# ---------------------------------------------------------------------------
# Ascend select_experts patch helper
# ---------------------------------------------------------------------------


def _patch_ascend_select_experts(experts, expansion_factor: int, n_routed: int):
    """Replace ``select_experts`` on Ascend with grouped routing.

    ``AscendZeroExpertFusedMoE.forward()`` temporarily disables
    ``custom_routing_function`` and calls ``select_experts`` directly,
    so neither the router patch nor ``custom_routing_function`` takes
    effect.  This helper replaces ``select_experts`` at the instance
    level so grouped routing runs inside the Ascend forward.
    """

    def grouped_select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = _grouped_routing(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expansion_factor=expansion_factor,
            n_routed_experts=n_routed,
            e_score_correction_bias=experts.e_score_correction_bias,
        )
        if experts.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * experts.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    experts.select_experts = grouped_select_experts


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


@register_patch(target="vllm.model_executor.models.longcat_flash")
def patch_longcat_flash_grouped_routing(module):
    """Monkey-patch vLLM ``longcat_flash`` for Grouped Routing support.

    Changes
    -------
    1. **FlashConfig** — accepts ``use_group_routing`` and
       ``expert_expansion_factor`` from the HF config.
    2. **LongcatMoe** — injects grouped routing.  Three code paths:

       *zero_expert_type is None*
           Sets ``custom_routing_function`` → ``CustomRoutingRouter``.
       *zero_expert_type is set, GPU*
           Patches ``ZeroExpertRouter._compute_routing`` directly (the
           router factory skips ``CustomRoutingRouter`` in this case).
       *Ascend NPU*
           Replaces ``select_experts`` on the ``AscendZeroExpertFusedMoE``
           instance (its ``forward()`` bypasses both of the above).
    """

    # ---- 1. FlashConfig ----

    original_config_init = module.FlashConfig.__init__

    def patched_config_init(
        self,
        use_group_routing: bool = False,
        expert_expansion_factor: int = 1,
        **kwargs,
    ):
        original_config_init(self, **kwargs)
        self.use_group_routing = use_group_routing
        self.expert_expansion_factor = expert_expansion_factor

    module.FlashConfig.__init__ = patched_config_init
    patch_logger.info("[longcat_flash] Patched FlashConfig for grouped routing fields")

    # ---- 2. LongcatMoe ----

    original_moe_init = module.LongcatMoe.__init__

    def patched_moe_init(
        self,
        config,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        original_moe_init(
            self,
            config=config,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            enable_eplb=enable_eplb,
        )

        use_group_routing = getattr(config, "use_group_routing", False)
        expansion_factor = getattr(config, "expert_expansion_factor", 1)

        if not (use_group_routing and expansion_factor > 1):
            return

        # n_routed_experts = N + Z (already computed by LongcatRouter)
        n_routed = self.router.n_routed_experts
        zero_expert_type = getattr(config, "zero_expert_type", None)
        is_ascend = hasattr(self.experts, "_temporarily_set_attrs")

        if zero_expert_type is not None and not is_ascend:
            # ---- Path A: GPU + zero experts ----
            # Router factory created a ZeroExpertRouter which ignores
            # custom_routing_function → patch _compute_routing directly.
            # NOTE: skipped on Ascend because AscendZeroExpertFusedMoE.forward()
            # calls select_experts directly (handled by Path C below), never
            # invoking ZeroExpertRouter._compute_routing.
            _patch_zero_expert_router(self.experts, expansion_factor, n_routed)
            patch_logger.info(
                "[longcat_flash] Enabled grouped routing on %s (ZeroExpertRouter path): "
                "expansion_factor=%d, n_routed_experts=%d, top_k=%d, zero_expert_type=%s",
                prefix or "LongcatMoe",
                expansion_factor,
                n_routed,
                top_k,
                zero_expert_type,
            )
        elif zero_expert_type is None:
            # ---- Path B: no zero expert (any device) ----
            # Router factory will create CustomRoutingRouter.
            self.experts.custom_routing_function = (
                lambda hidden_states, gating_output, topk, renormalize, **kw: (
                    _grouped_routing(
                        hidden_states,
                        gating_output,
                        topk,
                        renormalize,
                        expansion_factor,
                        n_routed,
                        e_score_correction_bias=(
                            self.router.e_score_correction_bias
                        ),
                    )
                )
            )
            patch_logger.info(
                "[longcat_flash] Enabled grouped routing on %s (CustomRoutingRouter path): "
                "expansion_factor=%d, n_routed_experts=%d, top_k=%d",
                prefix or "LongcatMoe",
                expansion_factor,
                n_routed,
                top_k,
            )

        # ---- Path C: Ascend NPU ----
        # AscendZeroExpertFusedMoE.forward() temporarily disables
        # custom_routing_function and calls select_experts directly,
        # bypassing both Path A's router patch and Path B's custom_routing_function.
        # This applies regardless of zero_expert_type.
        if is_ascend:
            _patch_ascend_select_experts(self.experts, expansion_factor, n_routed)
            patch_logger.info(
                "[longcat_flash] Enabled grouped routing on %s (Ascend select_experts path): "
                "expansion_factor=%d, n_routed_experts=%d, top_k=%d",
                prefix or "LongcatMoe",
                expansion_factor,
                n_routed,
                top_k,
            )

    module.LongcatMoe.__init__ = patched_moe_init

    patch_logger.info("[longcat_flash] Grouped Routing monkey patch applied")
