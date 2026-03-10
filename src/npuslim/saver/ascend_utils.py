"""
Utilities for extracting quantization state from models for Ascend deployment.
"""

from typing import Dict, Set, List
import torch.nn as nn


def get_all_tensor_names(model: nn.Module) -> Set[str]:
    """Get all tensor names (parameters + buffers) in the model."""
    names = set()
    for name, _ in model.named_parameters():
        names.add(name)
    for name, _ in model.named_buffers():
        names.add(name)
    return names


def get_quantized_layer_names(model: nn.Module, quant_layer_types: List[str]) -> Set[str]:
    """
    Get names of layers that have been replaced with quantized versions.

    Args:
        model: PyTorch model
        quant_layer_types: List of quantized layer class names (e.g., ["GPTQQuantLinear"])

    Returns:
        Set of module names that are quantized
    """
    quantized_names = set()
    for name, module in model.named_modules():
        if type(module).__name__ in quant_layer_types:
            quantized_names.add(name)
    return quantized_names


def build_tensor_quant_status(
    model: nn.Module,
    quantized_layer_names: Set[str],
    quant_type: str,
    has_offset: bool = True,
    include_g_idx: bool = False,
) -> Dict[str, str]:
    """
    Build per-tensor quantization status mapping.

    Checks both parameters and buffers to detect quantized tensors.
    Supports both Ascend format (weight, weight_scale, weight_offset) and
    GPTQ format (qweight, scales, qzeros, g_idx).

    Args:
        model: The quantized model
        quantized_layer_names: Set of layer base names that are quantized
        quant_type: Quantization type string (e.g., "W4A16", "W8A8_dynamic")
        has_offset: Whether quantized layers have weight_offset parameter
        include_g_idx: Whether to include g_idx tensor status

    Returns:
        Dict mapping tensor names to their quantization status
    """
    status = {}
    float_type = "FLOAT"

    # Process all parameters
    for name, _ in model.named_parameters():
        status[name] = _get_single_tensor_status(
            name, quantized_layer_names, quant_type, float_type, has_offset, include_g_idx
        )

    # Process all buffers (includes quantized weights)
    for name, _ in model.named_buffers():
        if name not in status:
            status[name] = _get_single_tensor_status(
                name, quantized_layer_names, quant_type, float_type, has_offset, include_g_idx
            )

    return status


def _get_single_tensor_status(
    name: str,
    quantized_layer_names: Set[str],
    quant_type: str,
    float_type: str,
    has_offset: bool,
    include_g_idx: bool,
) -> str:
    """Get the quantization status for a single tensor."""
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        module_name, tensor_type = parts
    else:
        module_name = ""
        tensor_type = parts[0]

    if module_name in quantized_layer_names:
        # Support both Ascend and GPTQ format tensor names
        if tensor_type in ("weight", "qweight"):
            return quant_type
        elif tensor_type in ("weight_scale", "scales"):
            return quant_type
        elif tensor_type in ("weight_offset", "qzeros") and has_offset:
            return quant_type
        elif tensor_type == "g_idx" and include_g_idx:
            return quant_type
        else:
            return float_type
    else:
        return float_type
