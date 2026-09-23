"""Patch fused_moe expert params mapping for W4A16 packed MoE params.

vLLM's ``FusedMoE.make_expert_params_mapping`` produces param-name prefixes
``experts.w13_`` / ``experts.w2_``, and the stock loader appends the checkpoint
chunk suffix (``.weight`` / ``.weight_scale`` / ``.weight_offset``), yielding
``experts.w13_weight`` / ``experts.w2_weight`` etc.

The vllm-ascend W4A16 FusedMoE method instead creates params named
``w13_weight_packed`` / ``w2_weight_packed`` (plus ``*_weight_scale`` and
``*_weight_offset``). For per-expert 2D checkpoints the standard mapping
therefore produces ``experts.w2_weight`` which does not exist in
``params_dict``, raising ``KeyError`` during model loading.

This patch wraps ``fused_moe_make_expert_params_mapping`` as bound in
``vllm.model_executor.models.deepseek_v2`` and, when the model uses ``_packed``
MoE params, expands each mapping entry into suffix-specific entries ordered as
[weight_scale, weight_offset, weight_packed] so the stock loader routes
per-expert 2D checkpoint weights into the packed 3D params.
"""

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch


def _uses_packed_moe(model) -> bool:
    for name, _ in model.named_parameters():
        if "experts." in name and "_packed" in name:
            return True
    return False


@register_patch(
    target="vllm.model_executor.models.deepseek_v2",
    condition=package_version_range("vllm", min_version="0.1.0"),
)
def patch_packed_expert_mapping(module):
    """Make expert params mapping W4A16-packed aware for DeepSeek-style models."""
    original = module.fused_moe_make_expert_params_mapping

    def patched(model, *args, **kwargs):
        mappings = original(model, *args, **kwargs)
        if not _uses_packed_moe(model):
            return mappings

        out = []
        for param_name, weight_name, expert_id, shard_id in mappings:
            # weight_name looks like "experts.0.gate_proj." (trailing dot).
            base = weight_name[:-1] if weight_name.endswith(".") else weight_name
            if "w13_" in param_name:
                p = "experts.w13_weight"
            else:
                p = "experts.w2_weight"
            # Order matters: the packed entry's weight_name
            # ("...proj.weight") is a substring of the scale/offset chunk
            # names, so the suffix-specific entries must come first.
            out.append((p + "_scale", base + ".weight_scale", expert_id, shard_id))
            out.append((p + "_offset", base + ".weight_offset", expert_id, shard_id))
            out.append((p + "_packed", base + ".weight", expert_id, shard_id))

        patch_logger.info(
            "[packed-moe] Expanded expert params mapping for W4A16 packed MoE "
            "(%d -> %d entries)", len(mappings), len(out)
        )
        return out

    module.fused_moe_make_expert_params_mapping = patched
    patch_logger.info(
        "Patched fused_moe_make_expert_params_mapping for W4A16 packed MoE"
    )
