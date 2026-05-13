#!/usr/bin/env python3
"""
Convert GPTQ-quantized models from GPU format to NPU (Ascend) format.

This script converts models quantized with standard GPTQ (GPU format) to the format
required by vLLM-Ascend for NPU deployment (Ascend format).

## Format Conversion Details

### Input (GPU Format):
- qweight: int32 packed weights [infeatures // 8, outfeatures]
- qzeros: int32 packed zero points [num_groups, outfeatures // 8]
- scales: float16/bfloat16 scales [num_groups, outfeatures]
- g_idx: int32 group indices [infeatures]
- config.json contains quantization_config with bits, group_size, etc.

### Output (NPU/Ascend Format - Column-wise Packing):
- weight: int32 packed weights [outfeatures, infeatures // 8] (8 int4 per int32, packed along input dim)
- weight_scale: bfloat16 scales [outfeatures, num_groups]
- weight_offset: bfloat16 offsets [outfeatures, num_groups]
- quantization_config is REMOVED from config.json (to prevent HF/vLLM from loading as standard GPTQ)
- quant_model_description.json is added for vLLM-Ascend

## Usage Examples

Basic conversion:
    python tools/convert/gptq_gpu_to_npu.py \
        --input /path/to/gpu_model \
        --output /path/to/npu_model

With verbose logging:
    python tools/convert/gptq_gpu_to_npu.py \
        --input /path/to/gpu_model \
        --output /path/to/npu_model \
        --verbose

## How It Works

1. Reads quantization parameters (bits, group_size) from config.json's quantization_config
2. Loads the GPTQ model with transformers
3. Unpacks int32 packed weights to individual 4-bit values
4. Converts from unsigned [0,15] to signed [-8,7] representation
5. Repacks into Ascend format (int32 with 8 int4 per int32 along input dim, column-wise)
6. Transforms scales/zeros to weight_scale/weight_offset format
7. Removes quantization_config from config.json
8. Adds ascend_quant_config and generates quant_model_description.json

## Requirements

- PyTorch
- Transformers
- NPUSlim (for GPTQQuantLinear class)
- loguru, tqdm

## Notes

- Only 4-bit quantization is supported for Ascend format
- The conversion process runs on CPU by default (recommended for large models)
- Output includes quant_model_description.json required by vLLM-Ascend
- The original quantization_config is removed to prevent HF/vLLM from trying to load it as standard GPTQ
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

# Add src to path for importing npuslim modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQQuantLinear


def load_quant_config(model_path: str) -> Dict[str, Any]:
    """
    Load quantization configuration from model's config.json.

    Args:
        model_path: Path to the model directory

    Returns:
        Dict with bits, group_size, and other quantization parameters

    Raises:
        ValueError: If quantization_config is not found or invalid
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    quant_config = config.get("quantization_config")
    if quant_config is None:
        raise ValueError(
            "No quantization_config found in config.json. "
            "Is this a GPTQ-quantized model?"
        )

    # Extract required parameters
    bits = quant_config.get("bits")
    group_size = quant_config.get("group_size")
    desc_act = quant_config.get("desc_act", quant_config.get("desc_act", True))
    static_groups = quant_config.get("static_groups", True)
    sym = quant_config.get("sym", True)

    if bits is None or group_size is None:
        raise ValueError(
            f"Missing required fields in quantization_config: {quant_config}"
        )

    logger.info(f"Loaded quant config: bits={bits}, group_size={group_size}, "
                f"desc_act={desc_act}, sym={sym}")

    return {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "static_groups": static_groups,
        "sym": sym,
    }


def unpack_gptq_weights(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack GPTQ format weights to float tensor.

    Args:
        qweight: [infeatures // 8, outfeatures] int32 packed weights
        qzeros: [num_groups, outfeatures // 8] int32 packed zeros
        scales: [num_groups, outfeatures] float32 scales
        g_idx: [infeatures] int32 group indices
        bits: quantization bits (2, 3, 4, or 8)
        group_size: size of each quantization group

    Returns:
        weight: [outfeatures, infeatures] int32 unpacked weights (unsigned 0-15 for 4-bit)
        zeros: [num_groups, outfeatures] float32 zero points
        scales: [num_groups, outfeatures] float32 scales
    """
    assert bits == 4, "Only 4-bit conversion is currently supported"

    infeatures_div8, outfeatures = qweight.shape
    infeatures = infeatures_div8 * 8
    num_groups = scales.shape[0]

    # Unpack weights from int32
    # Each int32 contains 8 4-bit values
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0).to(qweight.device)

    # Unpack qweight: [infeatures // 8, outfeatures] -> [infeatures, outfeatures]
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16)
    weight = torch.bitwise_and(weight, (2 ** bits) - 1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    # Unpack qzeros
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16)
    zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)
    zeros = zeros + 1  # GPTQ stores zeros - 1
    zeros = zeros.reshape(zeros.shape[0], zeros.shape[1] * zeros.shape[2])

    return weight, zeros.float(), scales.float()


def pack_ascend_weights(
    weight: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack weights into Ascend NPU format (column-wise packing).

    Args:
        weight: [infeatures, outfeatures] int32 unpacked weights (unsigned 0-15)
        zeros: [num_groups, outfeatures] float32 zero points
        scales: [num_groups, outfeatures] float32 scales
        g_idx: [infeatures] int32 group indices
        bits: quantization bits (4 only for Ascend)
        group_size: size of each quantization group

    Returns:
        packed_weight: [outfeatures, infeatures // 8] int32 (8 int4 packed per int32, column-wise)
        weight_scale: [outfeatures, num_groups] bfloat16
        weight_offset: [outfeatures, num_groups] bfloat16

    vLLM-Ascend expects (column-wise packing):
        weight:       [outfeatures, infeatures // 8] as int32 (packed int4 along input dim)
        weight_scale: [outfeatures, num_groups] as bfloat16
        weight_offset: [outfeatures, num_groups] as bfloat16

    Packing pattern: 8 int4 values per int32, packed along input dimension (columns).
        int32[i,j] = int4[i, 8*j] | (int4[i, 8*j+1] << 4) | ... | (int4[i, 8*j+7] << 28)
    """
    assert bits == 4, "Ascend only supports 4-bit quantization"

    infeatures, outfeatures = weight.shape
    num_groups = scales.shape[0]
    pack_factor = 8  # 8 int4 values per int32

    assert infeatures % pack_factor == 0, (
        f"Ascend format requires infeatures {infeatures} divisible by {pack_factor}"
    )

    # Transpose to [outfeatures, infeatures] for processing
    weight_t = weight.t().contiguous()

    # Convert to signed int4 range [-8, 7] from unsigned [0, 15]
    # In Ascend format: signed_offset = 8 for 4-bit
    signed_offset = 2 ** (bits - 1)  # 8 for 4-bit
    weight_signed = weight_t.to(torch.int32) - signed_offset
    weight_signed = weight_signed.clamp(-signed_offset, signed_offset - 1)

    # Convert signed int4 [-8, 7] to unsigned [0, 15] for packing
    weight_unsigned = (weight_signed + signed_offset).to(torch.uint8)

    # Pack 8 int4 values into int32 along input dimension (columns)
    # Shape: [outfeatures, infeatures] -> [outfeatures, infeatures // 8]
    packed_in = infeatures // pack_factor
    packed_weight = torch.zeros((outfeatures, packed_in), dtype=torch.int32)

    for i in range(pack_factor):
        # Each group of 8 consecutive columns packs into one int32
        # Col j*8+k goes to packed[:, j] at bits [4*k, 4*k+3]
        col_indices = torch.arange(i, infeatures, pack_factor)
        packed_weight |= (weight_unsigned[:, col_indices].to(torch.int32) << (bits * i))

    # Convert scales to NPU format
    # scales: [num_groups, outfeatures] -> [outfeatures, num_groups]
    weight_scale = scales.t().contiguous().to(torch.bfloat16)

    # For Ascend, weight_offset should be zeros (as the signed shift is
    # already integrated into the packed weight values)
    weight_offset = torch.zeros_like(weight_scale, dtype=torch.bfloat16)

    return packed_weight.contiguous(), weight_scale, weight_offset


def convert_layer_to_ascend(
    module: nn.Module,
    layer_name: str,
    bits: int = 4,
    group_size: int = 128,
) -> Optional[GPTQQuantLinear]:
    """
    Convert a single GPTQ linear layer from GPU to NPU format.

    Args:
        module: The GPTQ linear layer module (with qweight, qzeros, scales, g_idx)
        layer_name: Name of the layer for logging
        bits: Quantization bits
        group_size: Group size for quantization

    Returns:
        New GPTQQuantLinear layer in Ascend format, or None if conversion fails
    """
    if not hasattr(module, 'qweight'):
        logger.warning(f"Layer {layer_name} does not have qweight, skipping")
        return None

    # Get dimensions from existing layer
    infeatures_div8, outfeatures = module.qweight.shape
    infeatures = infeatures_div8 * (32 // bits)

    logger.debug(f"Converting {layer_name}: in={infeatures}, out={outfeatures}")

    # Extract tensors
    qweight = module.qweight.data
    qzeros = module.qzeros.data
    scales = module.scales.data
    g_idx = module.g_idx.data if hasattr(module, 'g_idx') else None

    if g_idx is None:
        # Create default g_idx if not present
        g_idx = torch.arange(infeatures, dtype=torch.int32) // group_size

    # Unpack GPU format
    weight, zeros, scales_fp = unpack_gptq_weights(
        qweight, qzeros, scales, g_idx, bits, group_size
    )

    # Pack into Ascend format
    packed_weight, weight_scale, weight_offset = pack_ascend_weights(
        weight, zeros, scales_fp, g_idx, bits, group_size
    )

    # Create new layer in Ascend format
    new_layer = GPTQQuantLinear(
        bits=bits,
        group_size=group_size,
        infeatures=infeatures,
        outfeatures=outfeatures,
        bias=hasattr(module, 'bias') and module.bias is not None,
        weight_dtype=scales.dtype,
        backend="npu",
    )

    # Copy packed weights
    new_layer.weight = nn.Parameter(packed_weight, requires_grad=False)
    new_layer.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
    new_layer.weight_offset = nn.Parameter(weight_offset, requires_grad=False)

    # Copy bias if present
    if new_layer.bias is not None and hasattr(module, 'bias') and module.bias is not None:
        new_layer.bias.data = module.bias.data.clone()

    return new_layer


def find_gptq_layers(model: nn.Module) -> dict:
    """Find all GPTQ quantized layers in the model."""
    gptq_layers = {}
    for name, module in model.named_modules():
        # Check if it looks like a GPTQ layer (has qweight)
        if hasattr(module, 'qweight') and hasattr(module, 'qzeros'):
            gptq_layers[name] = module
    return gptq_layers


def load_gptq_model(model_path: str, device: str = "cpu"):
    """
    Load a GPTQ-quantized model from disk.

    This function loads a model that was quantized and saved with GPTQQuantLinear layers.
    It handles the custom layer structure by:
    1. Loading config to get quantization parameters
    2. Loading the model with transformers (creates standard Linear layers initially)
    3. Replacing appropriate Linear layers with GPTQQuantLinear
    4. Loading the quantized state dict

    Args:
        model_path: Path to the GPTQ model directory
        device: Device to load the model on (default: "cpu")

    Returns:
        Tuple of (model, config)
    """
    from safetensors.torch import load_file

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Get quantization parameters from config
    quant_cfg = config.quantization_config
    bits = quant_cfg["bits"]
    group_size = quant_cfg["group_size"]

    logger.info(f"Loading GPTQ model with bits={bits}, group_size={group_size}")

    # Load model architecture without loading weights yet
    # We use low_cpu_mem_usage=True for efficient loading
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Find all Linear layers that need to be replaced with GPTQQuantLinear
    # by checking which ones have qweight in the state dict
    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        # Single safetensors file
        weight_map = {}

    # Identify layers that have GPTQ weights (qweight)
    gptq_layer_names = set()

    if weight_map:
        # Use index file weight map
        for key in weight_map.keys():
            if ".qweight" in key:
                # Extract layer name (e.g., "model.layers.0.mlp.experts.0.gate_proj")
                layer_name = key.rsplit(".qweight", 1)[0]
                gptq_layer_names.add(layer_name)
    else:
        # No index file - scan safetensors files directly
        safetensors_files = sorted(Path(model_path).glob("model-*.safetensors"))
        if not safetensors_files:
            single_file = Path(model_path) / "model.safetensors"
            if single_file.exists():
                safetensors_files = [single_file]

        for sf_file in safetensors_files:
            with safe_open(sf_file, framework="pt") as f:
                for key in f.keys():
                    if ".qweight" in key:
                        layer_name = key.rsplit(".qweight", 1)[0]
                        gptq_layer_names.add(layer_name)

    logger.info(f"Found {len(gptq_layer_names)} GPTQ layers to replace")

    # Replace Linear layers with GPTQQuantLinear (GPU format)
    for layer_name in gptq_layer_names:
        # Navigate to the parent module
        *parent_path, leaf_name = layer_name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)

        old_layer = getattr(parent, leaf_name)

        # Create GPTQQuantLinear with same dimensions
        infeatures = old_layer.in_features
        outfeatures = old_layer.out_features
        has_bias = old_layer.bias is not None

        # Detect weight dtype from config
        weight_dtype = torch.float16
        if hasattr(config, 'dtype'):
            if config.dtype == "bfloat16":
                weight_dtype = torch.bfloat16
            elif config.dtype == "float32":
                weight_dtype = torch.float32

        new_layer = GPTQQuantLinear(
            bits=bits,
            group_size=group_size,
            infeatures=infeatures,
            outfeatures=outfeatures,
            bias=has_bias,
            weight_dtype=weight_dtype,
            backend="cpu",
        )

        setattr(parent, leaf_name, new_layer)

    # Load state dict from safetensors files
    safetensors_files = sorted(Path(model_path).glob("model-*.safetensors"))
    if not safetensors_files:
        # Try single file
        single_file = Path(model_path) / "model.safetensors"
        if single_file.exists():
            safetensors_files = [single_file]

    logger.info(f"Loading {len(safetensors_files)} safetensors files")

    for sf_file in tqdm(safetensors_files, desc="Loading weights"):
        state_dict = load_file(sf_file, device=device)
        # Load into model (strict=False to handle any missing/unexpected keys)
        model.load_state_dict(state_dict, strict=False)

    # Move model to device
    model = model.to(device)
    model.eval()

    logger.info("GPTQ model loaded successfully")

    return model, config


def build_quant_description(model, bits: int, group_size: int) -> dict:
    """Build quant_model_description.json content."""
    description = {
        "version": "1.0.0",
        "model_quant_type": f"W{bits}A16",
        "group_size": group_size,
    }

    # Add per-tensor status
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module, 'weight_scale'):
            # This is a quantized layer - keep dots in name
            description[f"{name}.weight"] = f"W{bits}A16"
            description[f"{name}.weight_scale"] = "FLOAT"
            if hasattr(module, 'weight_offset'):
                description[f"{name}.weight_offset"] = "FLOAT"
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            # Regular float layer - keep dots in name
            description[f"{name}.weight"] = "FLOAT"

    return description


def convert_model(
    input_path: str,
    output_path: str,
    device: str = "cpu",
):
    """
    Convert a full GPTQ model from GPU format to NPU format.

    Args:
        input_path: Path to input GPU-format GPTQ model
        output_path: Path to save NPU-format model
        device: Device to use for conversion
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load quantization config from model
    quant_cfg = load_quant_config(input_path)
    bits = quant_cfg["bits"]
    group_size = quant_cfg["group_size"]

    logger.info(f"Converting with bits={bits}, group_size={group_size}")

    # Load model using safetensors
    model, config = load_gptq_model(input_path, device)

    # Find all GPTQ layers
    gptq_layers = find_gptq_layers(model)
    logger.info(f"Found {len(gptq_layers)} GPTQ layers to convert")

    if len(gptq_layers) == 0:
        logger.error("No GPTQ layers found in model. Is this a GPTQ-quantized model?")
        return

    # Convert each layer
    for name, module in tqdm(gptq_layers.items(), desc="Converting layers"):
        new_layer = convert_layer_to_ascend(module, name, bits, group_size)

        if new_layer is None:
            continue

        # Find parent module and replace
        *parent_path, leaf_name = name.split('.')
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)

        setattr(parent, leaf_name, new_layer)

    logger.info("Layer conversion complete")

    # Convert any remaining float16 tensors to bfloat16
    # (some layers like embed_tokens, lm_head are not quantized and remain float16)
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.bfloat16)

    # Force dtype to bfloat16 since weight_scale/weight_offset are always bf16
    # NPU requires config dtype to match actual tensor dtypes
    model.config.dtype = torch.bfloat16

    # Update config for Ascend
    model.config.ascend_quant_config = {
        "model_quant_type": f"W{bits}A16",
        "group_size": group_size,
        "quant_layer_types": ["GPTQQuantLinear"],
        "include_g_idx": True,
        "has_offset": True,
    }

    # Remove GPU-specific quantization_config if present
    if hasattr(model.config, 'quantization_config'):
        delattr(model.config, 'quantization_config')

    # Save model
    logger.info(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Generate quant_model_description.json
    description = build_quant_description(model, bits, group_size)
    desc_path = output_path / "quant_model_description.json"
    with open(desc_path, "w") as f:
        json.dump(description, f, indent=2)
    logger.info(f"Generated {desc_path}")

    # Save tokenizer if exists
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        logger.info("Tokenizer saved")
    except Exception as e:
        logger.warning(f"Could not save tokenizer: {e}")

    logger.success(f"Conversion complete! Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPTQ-quantized models from GPU format to NPU (Ascend) format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input GPU-format GPTQ model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output NPU-format model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for conversion (default: cpu)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(lambda msg: print(msg, end=""), level=log_level)

    # Run conversion
    convert_model(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
