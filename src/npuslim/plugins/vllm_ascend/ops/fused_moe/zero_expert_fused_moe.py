"""Ascend OOT registration for ``ZeroExpertFusedMoE``.

Ascend registers an OOT replacement for ``FusedMoE`` but not for
``ZeroExpertFusedMoE``.  LongCat-Flash instantiates the latter, so it keeps
upstream zero-expert control flow while missing the Ascend runner and
Ascend-compatible routing helpers.

EP (MC2 / All2All / FusedMC2) path
-----------------------------------
For expert-parallel deployments the MC2 prepare step pads hidden_states to
``padded_num_tokens`` and slices by TP rank.  Routing must be computed on the
*prepared* tokens so that memoized top-k weights / ids match the prepared
layout.  This class overrides ``forward_impl`` to run::

    prepare -> route -> zero-expert filter -> apply -> finalize

keeping routing after prepare.

Non-EP (AllGather) path
------------------------
Without expert parallelism the prepare step does not change the token
dimension, so routing is computed in ``forward`` before delegating to the
parent ``forward_impl``.  The memoization trick from upstream works unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE,
)
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts as ascend_select_experts,
    zero_experts_compute as ascend_zero_experts_compute,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, FusedMoEResult

from npuslim.plugins.registry import package_version_range, register_patch

if TYPE_CHECKING:
    pass

# vllm-ascend < 0.19 uses ``global_num_experts``; >= 0.19 uses ``num_experts``.
import importlib.metadata
from packaging.version import Version

_USE_GLOBAL_NUM_EXPERTS = Version(importlib.metadata.version("vllm_ascend")) < Version("0.19")


def _expert_count_kwarg(count: int) -> dict:
    """Return ``{global_num_experts: count}`` or ``{num_experts: count}``."""
    if _USE_GLOBAL_NUM_EXPERTS:
        return {"global_num_experts": count}
    return {"num_experts": count}


_EP_COMM_TYPES = frozenset({
    MoECommType.MC2,
    MoECommType.FUSED_MC2,
    MoECommType.ALLTOALL,
})


@register_patch(
    registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendZeroExpertFusedMoE(ZeroExpertFusedMoE, AscendFusedMoE):
    """Ascend replacement for upstream ``ZeroExpertFusedMoE``.

    For EP (MC2/All2All): overrides ``forward_impl`` to compute routing AFTER
    prepare, so memoized values match the prepared token layout.

    For non-EP (AllGather): uses upstream memoization in ``forward`` (routing
    before forward_impl) because prepare doesn't change token dim.
    """

    def __init__(
        self,
        zero_expert_num: int,
        zero_expert_type: str,
        router,
        **kwargs,
    ) -> None:
        super().__init__(
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
            router=router,
            **kwargs,
        )

        def custom_routing_function(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            **_ignored,
        ):
            if (
                self._memoized_topk_weights is None
                or self._memoized_topk_ids is None
            ):
                raise RuntimeError(
                    "ZeroExpertFusedMoE: routing results not memoized. "
                    "Call select_experts first to compute routing."
                )
            return self._memoized_topk_weights, self._memoized_topk_ids

        self.custom_routing_function = custom_routing_function

    def _compute_zero_expert_result(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute zero experts with Ascend helpers and memoize filtered routing."""
        if (
            self._actual_zero_expert_num is None
            or self._actual_zero_expert_num <= 0
            or self._actual_zero_expert_type is None
        ):
            self._memoized_topk_weights = topk_weights
            self._memoized_topk_ids = topk_ids
            return None

        topk_weights = topk_weights.to(hidden_states.dtype)

        filtered_topk_ids, filtered_topk_weights, zero_expert_result = (
            ascend_zero_experts_compute(
                expert_indices=topk_ids.clone(),
                expert_scales=topk_weights.clone(),
                num_experts=self.logical_num_experts,
                zero_expert_type=self._actual_zero_expert_type,
                hidden_states=hidden_states,
            )
        )
        self._memoized_topk_weights = filtered_topk_weights
        self._memoized_topk_ids = filtered_topk_ids
        return zero_expert_result.to(hidden_states.dtype)

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts using the Ascend routing path."""
        return ascend_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            **_expert_count_kwarg(router_logits.shape[-1]),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with Ascend routing and external zero-expert fusion."""
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        is_ep = moe_comm_type in _EP_COMM_TYPES

        if is_ep:
            # EP: bypass AscendFusedMoE.forward() → runner → moe_forward op
            # chain.  That indirect path goes through DefaultMoERunner.forward()
            # which dispatches via torch.ops.vllm.moe_forward, introducing an
            # extra layer of custom-op dispatch and layer lookup that can
            # deadlock EP ranks.  Call forward_impl directly instead — it
            # already handles the full EP flow (prepare → route → apply →
            # finalize).
            self.ensure_moe_quant_config_init()
            return self.forward_impl(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # Non-EP: compute routing before forward_impl.  AllGather prepare
        # does not change the token dimension, so memoized values match.
        temp_attrs = {
            "custom_routing_function": None,
        }
        if self._router is not None:
            temp_attrs["e_score_correction_bias"] = (
                self._router.e_score_correction_bias
            )

        with self._temporarily_set_attrs(**temp_attrs):
            topk_weights, topk_ids = self.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        zero_expert_result = self._compute_zero_expert_result(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        router_logits_sliced = router_logits[..., : self.logical_num_experts]

        try:
            fused_out = AscendFusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits_sliced,
            )
        finally:
            self._memoized_topk_weights = None
            self._memoized_topk_ids = None

        if zero_expert_result is not None:
            fused_out = fused_out + zero_expert_result

        return fused_out

    # ------------------------------------------------------------------
    # forward_impl — EP override
    # ------------------------------------------------------------------

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        return_with_event: bool = False,
    ) -> torch.Tensor | FusedMoEResult:
        assert self.quant_method is not None

        moe_comm_type = _EXTRA_CTX.moe_comm_type
        is_ep = moe_comm_type in _EP_COMM_TYPES

        if not is_ep:
            # Non-EP: parent handles it with memoized routing from forward().
            return AscendFusedMoE.forward_impl(
                self, hidden_states, router_logits, return_with_event
            )

        # ---- EP + zero experts: routing AFTER prepare ----

        forward_context = get_forward_context()
        if (
            self.enable_npugraph_ex_static_kernel
            and forward_context.all_moe_layers
        ):
            moe_layer_index = forward_context.moe_layer_index % len(
                forward_context.all_moe_layers
            )
            forward_context.moe_layer_index = moe_layer_index

        # The normal runner → moe_forward custom-op chain increments
        # moe_layer_index between layers.  Our bypass skips that chain,
        # so we must increment it here so that each MoE layer gets a
        # unique index for MC2/All2All communication handles.
        if hasattr(forward_context, "all_moe_layers") and forward_context.all_moe_layers:
            forward_context.moe_layer_index += 1

        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        # 1. Prepare: pad + TP-slice for EP
        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        prepared_hs = prepare_output.hidden_states
        prepared_rl = prepare_output.router_logits
        mc2_mask = prepare_output.mc2_mask
        padded_hs_shape = prepare_output.padded_hidden_states_shape
        pertoken_scale = prepare_output.pertoken_scale

        # 2. Compute routing on PREPARED hidden_states
        e_score_bias = (
            self._router.e_score_correction_bias
            if self._router is not None
            else None
        )
        topk_weights, topk_ids = ascend_select_experts(
            hidden_states=prepared_hs,
            router_logits=prepared_rl,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=None,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=e_score_bias,
            **_expert_count_kwarg(prepared_rl.shape[-1]),
        )

        # 3. Compute zero-expert result on PREPARED hidden_states.
        zero_expert_result = self._compute_zero_expert_result(
            hidden_states=prepared_hs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # 4. Slice prepared router_logits to exclude zero experts
        prepared_rl_sliced = prepared_rl[..., : self.logical_num_experts]

        # 5. Run quant_method.apply
        try:
            fused_experts_results = self.quant_method.apply(
                layer=self,
                x=prepared_hs,
                router_logits=prepared_rl_sliced,
                pertoken_scale=pertoken_scale,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                **_expert_count_kwarg(self.global_num_experts),
                expert_map=self._expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_force_load_balance=enable_force_load_balance,
                log2phy=self.log2phy,
                global_redundant_expert_num=self.global_redundant_expert_num,
                mc2_mask=mc2_mask,
            )
        finally:
            self._memoized_topk_weights = None
            self._memoized_topk_ids = None

        # 6. Combine zero-expert output before finalize
        routed_out = fused_experts_results.routed_out
        if zero_expert_result is not None:
            routed_out = routed_out + zero_expert_result

        if self.dynamic_eplb:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type should not be None when "
                "dynamic_eplb is enabled."
            )
            local_load = (
                expert_tokens
                if group_list_type == 1
                else torch.cat(
                    [expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]]
                )
            )
            if self.multi_stage:
                cur_iter = torch.remainder(self.load_counter, self.num_iter)
                self.moe_load.index_add_(
                    dim=0,
                    index=cur_iter,
                    source=local_load.to(
                        torch.int32, non_blocking=True
                    ).view(1, -1),
                )
                self.load_counter.add_(1)
            else:
                self.moe_load.add_(local_load)

        # 7. Finalize routed output (all-gather + unpad)
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=routed_out,
            reduce_results=self.reduce_results,
            padded_hidden_states_shape=padded_hs_shape,
        )

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
            )
        return routed_out
