"""Ascend OOT registration for ``ZeroExpertFusedMoE``.

Ascend registers an OOT replacement for ``FusedMoE`` but not for
``ZeroExpertFusedMoE``. LongCat-Flash instantiates the latter, so it keeps
upstream zero-expert control flow while missing the Ascend runner and
Ascend-compatible routing helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE,
)
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts as ascend_select_experts,
    zero_experts_compute as ascend_zero_experts_compute,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from npuslim.plugins.registry import package_version_range, register_patch

if TYPE_CHECKING:
    import torch


@register_patch(
    registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE"),
    condition=package_version_range("vllm_ascend", max_version="0.18.1"),
)
class AscendZeroExpertFusedMoE(ZeroExpertFusedMoE, AscendFusedMoE):
    """Ascend replacement for upstream ``ZeroExpertFusedMoE``.

    The init order matters:
    1. ``ZeroExpertFusedMoE`` sets up router memoization state and zero-expert
       bookkeeping used by its custom ``forward``.
    2. Its ``super().__init__`` resolves to ``AscendFusedMoE`` via MRO, so
       the layer still gets the Ascend runner and quantization methods.
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
        """Compute zero experts with Ascend helpers and filter routing.

        We intentionally keep ``self.zero_expert_num/type`` at upstream
        placeholder values (0/None), because the current Ascend
        ``quant_method.apply()`` zero-expert branch can return a
        ``FusedExpertsResult`` and then fail on ``+= zero_expert_result``.
        Instead, we compute the zero-expert contribution here and stash
        filtered routing results for the second MoE pass.
        """
        if (
            self._actual_zero_expert_num is None
            or self._actual_zero_expert_num <= 0
            or self._actual_zero_expert_type is None
        ):
            self._memoized_topk_weights = topk_weights
            self._memoized_topk_ids = topk_ids
            return None

        # Match upstream zero_experts_compute_triton dtype behavior:
        # zero-expert outputs must stay in hidden_states.dtype, otherwise
        # the MoE branch is promoted to FP32 and later residual RMSNorm
        # receives mismatched dtypes (FP32 vs BF16).
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
        """Select experts using the Ascend routing path.

        ``LongcatRouter`` only provides logits and bias, so we call the
        Ascend selector directly instead of relying on an in-tree fused
        MoE router object.
        """

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
            global_num_experts=router_logits.shape[-1],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with Ascend routing and external zero-expert fusion."""

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
