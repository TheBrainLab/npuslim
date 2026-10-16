"""Router-dump patch for DeepseekV2/GLM-5.2 MoE (P1: MoE router consistency).

Env-gated: when ``NPUSLIM_ROUTER_DUMP`` is set, wraps ``DeepseekV2MoE.forward``
to record the **router top-k expert ids** for the first ``NPUSLIM_ROUTER_CALLS``
forward calls of every MoE layer, and writes a torch file mapping
``layer_index -> int64[P, k]`` (P = tokens processed by this rank).

Rationale
---------
The L0 "MoE router consistency" instrument (see
``npuslim-docs/GLM-5.2-量化评测落地方案1015.md`` §2) is the only diagnostic the
public benchmarks cannot provide: W4A8 quantises the routed experts while the
router stays BF16, so expert *selection* can drift via the activation
distribution without any accuracy metric moving.

Usage
-----
Set on the vLLM serve process (docker exec -e ...):
  NPUSLIM_ROUTER_DUMP=/tmp/router_l0.pt
  NPUSLIM_ROUTER_TOPK=8        (default 8)
  NPUSLIM_ROUTER_CALLS=1       (record first N forward calls per layer)
  NPUSLIM_ROUTER_RANK=0        (only this TP rank writes; default 0)

Then send the fixed prompt set and read the file from the container.
Compare across products (same rank, same prompts): top-1 agreement,
expert-usage KL, routing entropy.
"""
from __future__ import annotations

import os

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


@register_patch(
    target="vllm.model_executor.models.deepseek_v2",
    condition=package_version_range("vllm", min_version="0.1.0"),
)
def patch_deepseek_moe_router_dump(module):  # noqa: ANN001
    dump_path = os.environ.get("NPUSLIM_ROUTER_DUMP")
    if not dump_path:
        return  # no-op unless requested

    topk = int(os.environ.get("NPUSLIM_ROUTER_TOPK", "8"))
    max_calls = int(os.environ.get("NPUSLIM_ROUTER_CALLS", "1"))
    want_rank = int(os.environ.get("NPUSLIM_ROUTER_RANK", "0"))

    moe_cls = getattr(module, "DeepseekV2MoE", None)
    if moe_cls is None:
        patch_logger.warning("DeepseekV2MoE not found - router dump skipped")
        return

    orig_forward = moe_cls.forward
    state = {"mods": {}, "records": {}, "writes": 0}

    def _dump():
        import torch

        torch.save(
            {
                "topk": topk,
                "max_calls": max_calls,
                "rank": want_rank,
                "layers": state["records"],
            },
            dump_path,
        )
        state["writes"] += 1

    def wrapped(self, hidden_states):  # noqa: ANN001
        try:
            if _rank() == want_rank:
                mid = id(self)
                if mid not in state["mods"]:
                    state["mods"][mid] = len(state["mods"])
                idx = state["mods"][mid]
                rec = state["records"].setdefault(idx, [])
                if len(rec) < max_calls:
                    import torch

                    if not getattr(self.experts, "is_internal_router", False):
                        router_logits, _ = self.gate(hidden_states)
                    else:
                        router_logits = hidden_states
                    logits = router_logits.float()
                    bias = getattr(self.gate, "e_score_correction_bias", None)
                    if bias is not None:
                        logits = logits + bias.float()
                    tk = torch.topk(logits, k=topk, dim=-1).indices
                    rec.append(tk.detach().to("cpu"))
                    _dump()
        except Exception as exc:  # never break serving
            patch_logger.warning(f"router dump failed (ignored): {exc}")
        return orig_forward(self, hidden_states)

    moe_cls.forward = wrapped
    patch_logger.info(
        f"router dump enabled -> {dump_path} (topk={topk}, calls={max_calls}, rank={want_rank})"
    )
