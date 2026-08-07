"""Patch for vllm/model_executor/models/glm_moe_dsa.py

This module patches GlmMoeDsaForCausalLM.load_weights to handle W4A16
quantization where MoE expert weights are stored as fused 3D tensors
with weight/weight_scale/weight_offset suffixes.

Root Cause:
- NPUSlim outputs fused 3D naming: experts.gate_up_proj.weight [E,2I,H//8]
- vLLM FusedMoE's expert_params_mapping expects per-expert naming:
  experts.{i}.gate_proj.weight, which doesn't match the fused 3D format
- W4A16 quantization registers params with _scale/_offset suffixes that
  the standard load_weights doesn't handle for expert weights

Solution:
- For W4A16 expert weights: bypass expert_params_mapping and load directly
- For non-expert weights: use the original stacked_params_mapping logic
- Try multiple suffixes (_packed, "", _scale, _shape, _offset) to find params
"""

from collections.abc import Iterable
from typing import Any

import torch
from vllm.logger import init_logger

from npuslim.plugins.logging import patch_logger
from npuslim.plugins.registry import package_version_range, register_patch

target_logger = init_logger(__name__)

# Suffixes for quantized weight parameters
# Ascend (NPU) format: weight (int32), weight_scale (bf16), weight_offset (bf16)
# GPU GPTQ format: qweight (int32), qzeros (int32), scales (fp16), g_idx (int32)
# GPU W4A16 format: weight_packed, weight_scale, weight_shape, weight_offset
_NPU_WEIGHT_SUFFIXES = ["weight"]
_NPU_AUX_SUFFIXES = ["weight_scale", "weight_offset"]
_GPU_GPTQ_SUFFIXES = ["qweight", "qzeros", "scales", "g_idx"]
_W4A16_WEIGHT_SUFFIXES = ["_packed", ""]
_W4A16_AUX_SUFFIXES = ["_scale", "_shape", "_offset"]

# Suffixes that indicate a critical (required) parameter for quantized experts.
# If any of these are unloaded after weight loading, we raise an error.
_CRITICAL_SUFFIXES = (
    "weight", "qweight", "weight_scale", "scales",
)

# All suffixes to try when matching expert weights
_ALL_QUANT_SUFFIXES = (
    _NPU_WEIGHT_SUFFIXES + _NPU_AUX_SUFFIXES
    + _GPU_GPTQ_SUFFIXES
    + _W4A16_WEIGHT_SUFFIXES + _W4A16_AUX_SUFFIXES
)

# Suffixes to ignore when loading non-quantized weights
_IGNORE_SUFFIXES = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
)


def _is_quantized(params_dict: dict) -> bool:
    """Check if the model uses quantization for MoE experts.

    Detects both NPU (W4A16) and GPU (GPTQ) quantization formats.
    """
    for name in params_dict:
        if "experts." in name:
            # NPU W4A16: weight_scale, weight_offset
            if "weight_scale" in name or "weight_offset" in name:
                return True
            # GPU GPTQ: qweight, qzeros, scales, g_idx
            if any(s in name for s in _GPU_GPTQ_SUFFIXES):
                return True
    return False


def _is_expert_weight(name: str) -> bool:
    """Check if a checkpoint weight name corresponds to a MoE expert tensor."""
    return ".experts." in name and (
        ".gate_up_proj." in name
        or ".down_proj." in name
        or ".gate_proj." in name
        or ".up_proj." in name
    )


@register_patch(
    target="vllm.model_executor.models.glm_moe_dsa",
    condition=package_version_range("vllm", min_version="0.1.0"),
)
def patch_glm_moe_dsa_load_weights(module):
    """Patch GlmMoeDsaForCausalLM.load_weights to handle quantized MoE experts.

    Supports both NPU W4A16 (weight/weight_scale/weight_offset) and
    GPU GPTQ (qweight/qzeros/scales/g_idx) quantization formats.

    The version condition is permissive (min_version only). Runtime signature
    compatibility is checked inside this function: if load_weights signature
    changes in future vLLM versions, the patch is skipped with a warning.
    """

    # Find the model class that has load_weights
    model_cls = None
    for cls_name in ("GlmMoeDsaForCausalLM", "GlmMoeDsaModel"):
        cls = getattr(module, cls_name, None)
        if cls is not None and hasattr(cls, "load_weights"):
            model_cls = cls
            break

    if model_cls is None:
        patch_logger.warning(
            "Could not find GlmMoeDsa model class with load_weights, skipping patch"
        )
        return

    # Runtime signature compatibility check: verify load_weights accepts
    # a single iterable argument. If vLLM changes the signature, skip patch.
    import inspect

    sig = inspect.signature(model_cls.load_weights)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1 or params[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        patch_logger.warning(
            f"{model_cls.__name__}.load_weights signature {sig} is incompatible "
            f"with the W4A16 patch (expected 1 positional arg). Skipping patch."
        )
        return

    original_load_weights = model_cls.load_weights

    def patched_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())

        # Check if quantization is used (NPU W4A16 or GPU GPTQ)
        is_quantized = _is_quantized(params_dict)

        if not is_quantized:
            # Use original implementation for non-quantized models
            return original_load_weights(self, weights)

        # Quantization-specific loading logic
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MLA attention uses different projection names
        # Check if MLA params exist and add mappings
        if any("q_a_proj" in name for name in params_dict):
            # GLM-5 uses MLA: q_a_proj + q_b_proj instead of q_proj
            # kv_a_proj_with_mqa + kv_b_proj instead of k_proj/v_proj
            stacked_params_mapping = [
                # MLA doesn't stack q/k/v, so use minimal mapping
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        loaded_params: set[str] = set()

        # Try to get expert_params_mapping (may not exist for all model classes)
        expert_params_mapping = []
        if hasattr(self, "get_expert_mapping"):
            try:
                expert_params_mapping = self.get_expert_mapping()
            except Exception:
                pass

        for name, loaded_weight in weights:
            # Handle KV cache quantization scales
            if self.quant_config is not None:
                scale_name = self.quant_config.get_cache_scale(name)
                if scale_name is not None:
                    param = params_dict[scale_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    assert loaded_weight.numel() == 1, (
                        f"KV scale numel {loaded_weight.numel()} != 1"
                    )
                    loaded_weight = loaded_weight.squeeze()
                    weight_loader(param, loaded_weight)
                    loaded_params.add(scale_name)
                    continue

            # Handle expert weights (quantized: direct loading with suffix matching)
            if _is_expert_weight(name):
                # For fused 3D checkpoint (experts.gate_up_proj.qweight),
                # try direct name match first, then suffix-based matching
                param_found = False

                def _try_load(candidate, param, loaded_weight, **kwargs):
                    """Try to load weight into param, return True on success."""
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    try:
                        success = weight_loader(
                            param, loaded_weight, candidate,
                            return_success=True, **kwargs
                        )
                        if success:
                            loaded_params.add(candidate)
                            return True
                    except TypeError:
                        try:
                            weight_loader(param, loaded_weight, **kwargs)
                            loaded_params.add(candidate)
                            return True
                        except Exception as e:
                            target_logger.warning(f"Failed to load {candidate}: {e}")
                    return False

                # 1. Direct match (works for GPU GPTQ: qweight/qzeros/scales/g_idx
                #    and NPU W4A16: weight/weight_scale/weight_offset)
                if name in params_dict:
                    if not is_pp_missing_parameter(name, self):
                        param = params_dict[name]
                        param_found = _try_load(name, param, loaded_weight)
                    else:
                        param_found = True

                # 2. Suffix-based match (for GPU W4A16: weight -> weight_packed, etc.)
                if not param_found:
                    for suffix in _W4A16_WEIGHT_SUFFIXES + _W4A16_AUX_SUFFIXES:
                        if not suffix:
                            continue  # Already tried direct match
                        candidate = name + suffix
                        if candidate in params_dict:
                            if is_pp_missing_parameter(candidate, self):
                                param_found = True
                                break
                            param = params_dict[candidate]
                            param_found = _try_load(candidate, param, loaded_weight)
                            if param_found:
                                break

                # 3. Try expert_params_mapping (expanded naming: experts.0.gate_proj.weight)
                if not param_found and expert_params_mapping:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue

                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            param_found = True
                            break

                        # Direct match on mapped name
                        if name_mapped in params_dict:
                            param = params_dict[name_mapped]
                            param_found = _try_load(
                                name_mapped, param, loaded_weight,
                                shard_id=shard_id, expert_id=expert_id
                            )
                            if param_found:
                                break

                        # Suffix-based match on mapped name
                        for suffix in _W4A16_WEIGHT_SUFFIXES + _W4A16_AUX_SUFFIXES:
                            if not suffix:
                                continue  # Already tried direct match
                            candidate = name_mapped + suffix
                            if candidate in params_dict:
                                param = params_dict[candidate]
                                param_found = _try_load(
                                    candidate, param, loaded_weight,
                                    shard_id=shard_id, expert_id=expert_id
                                )
                                if param_found:
                                    break

                        if param_found:
                            break

                if param_found:
                    continue
                # Expert weight not found locally, skip
                continue

            # Handle stacked params (non-expert: q/k/v, gate/up, etc.)
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name:
                    continue  # Already handled above

                name_mapped = name.replace(weight_name, param_name)

                if name_mapped.endswith(_IGNORE_SUFFIXES) and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if name_mapped.endswith("scale"):
                    remapped = maybe_remap_kv_scale_name(name_mapped, params_dict)
                    if remapped is None:
                        continue
                    name_mapped = remapped
                if name_mapped not in params_dict:
                    # Try with quantization suffixes (NPU W4A16 + GPU GPTQ + GPU W4A16)
                    for suffix in _ALL_QUANT_SUFFIXES:
                        if not suffix:
                            continue  # Already tried direct match
                        name_with_suffix = name_mapped + suffix
                        if name_with_suffix in params_dict:
                            name_mapped = name_with_suffix
                            break
                    else:
                        continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                handled = True
                break

            if handled:
                continue

            # Handle remaining non-expert weights
            if name.endswith(_IGNORE_SUFFIXES) and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith("kv_scale"):
                remapped_kv_scale_name = name.replace(
                    ".kv_scale", ".attn.kv_scale"
                )
                if remapped_kv_scale_name not in params_dict:
                    target_logger.warning_once(
                        "Found kv scale in checkpoint (e.g. %s), "
                        "but not found expected name in model (e.g. %s). "
                        "kv-scale is not loaded.",
                        name,
                        remapped_kv_scale_name,
                    )
                    continue
                name = remapped_kv_scale_name

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Check for unloaded expert parameters after processing all weights.
        # Critical parameters (weight/qweight/weight_scale/scales) that are
        # missing indicate a checkpoint-model mismatch and will cause silent
        # inference errors if not caught here.
        unloaded_expert_params = []
        unloaded_critical = []
        for param_name in params_dict:
            if ".experts." not in param_name:
                continue
            if param_name in loaded_params:
                continue
            unloaded_expert_params.append(param_name)
            if any(param_name.endswith(s) for s in _CRITICAL_SUFFIXES):
                unloaded_critical.append(param_name)

        if unloaded_critical:
            raise RuntimeError(
                f"Failed to load {len(unloaded_critical)} critical expert "
                f"parameter(s) from checkpoint: {unloaded_critical[:10]}"
                + ("..." if len(unloaded_critical) > 10 else "")
            )
        if unloaded_expert_params:
            target_logger.warning(
                f"{len(unloaded_expert_params)} non-critical expert parameter(s) "
                f"were not loaded: {unloaded_expert_params[:10]}"
                + ("..." if len(unloaded_expert_params) > 10 else "")
            )

        return loaded_params

    model_cls.load_weights = patched_load_weights
    patch_logger.info(
        f"Patched {model_cls.__name__}.load_weights for W4A16 MoE support"
    )
