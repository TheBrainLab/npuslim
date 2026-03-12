"""W4A16 Linear quantization scheme for Ascend NPU.

This scheme is registered with vllm-ascend via @register_scheme decorator.
"""

import torch
import torch_npu
from typing import Any

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme
from vllm_ascend.utils import maybe_trans_nz


@register_scheme("W4A16", "linear")
class AscendW4A16LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A16.

    This scheme uses 4-bit quantized weights with 16-bit activations.
    Supports both per-channel and per-group quantization with int8 packed
    weights (2 int4 per int8).

    msmodelslim format:
    - Weight: [output_size // 2, input_size] (packed along output dim)
    - Scale: [output_size, num_groups] where num_groups = input_size // group_size

    After processing:
    - Weight: [input_size, output_size] (transposed for NPU API)
    - Scale: [num_groups, output_size] (transposed for NPU API)
    """

    def __init__(self) -> None:
        self.storage_pack_factor = 2  # 2 int4 per int8
        self.group_size = 128  # Default group size for per-group quantization

        # Try to get group_size from vllm config if available
        try:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            if hasattr(vllm_config, 'quant_config') and hasattr(vllm_config.quant_config, 'quant_description'):
                self.group_size = vllm_config.quant_config.quant_description.get("group_size", 128)
        except Exception:
            pass

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create weight parameters.

        For W4A16, weights are stored as int8 (packed int4).
        msmodelslim format: [output_size // 2, input_size] (output packed)
        """
        assert output_size % self.storage_pack_factor == 0, (
            f"Expecting `output_size` {output_size} "
            f"can be divided by `storage_pack_factor` {self.storage_pack_factor}"
        )

        params_dict: dict[str, Any] = {}
        # Weight shape: [output_size // 2, input_size] as int8
        # Each int8 contains 2 int4 values packed along output dimension
        params_dict["weight"] = torch.empty(
            output_size // self.storage_pack_factor, input_size, dtype=torch.int8
        )
        # Tell vLLM's weight_loader about the packed dimension
        # packed_dim=0 means output dimension (dim 0) is packed
        # packed_factor=2 means each element represents 2 packed values
        params_dict["_packed_dim"] = 0
        params_dict["_packed_factor"] = self.storage_pack_factor
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create per-channel quantization parameters.

        For W4A16 per-channel, scale and offset are [output_size, 1].
        """
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
    ) -> dict[str, Any]:
        """Create per-group quantization parameters.

        For W4A16 per-group, scale and offset are [output_size, input_size // group_size].

        Args:
            input_size: Input dimension of the linear layer.
            output_size: Output dimension of the linear layer.
            params_dtype: Data type for parameters.
            layer_type: Type of layer (e.g., "row" for RowParallelLinear).
        """
        assert input_size % self.group_size == 0, (
            f"Expecting `input_size` {input_size} "
            f"can be divided by `group_size` {self.group_size}"
        )

        num_groups = input_size // self.group_size
        params_dict: dict[str, Any] = {}
        params_dict["weight_scale"] = torch.empty(output_size, num_groups, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, num_groups, dtype=params_dtype)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Forward computation using W4A16 quantized matmul.

        Supports both per-channel and per-group quantization.
        Note: NPU requires x and antiquant_scale to have the same dtype.
        """
        # Cast scale and offset to match input dtype (NPU requirement)
        scale = layer.weight_scale.to(x.dtype)
        offset = layer.weight_offset.to(x.dtype) if layer.weight_offset is not None else None

        # Determine if per-group quantization is used
        # Per-group: scale shape is [num_groups, output_size]
        # Per-channel: scale shape is [output_size] or [1, output_size]
        use_per_group = hasattr(layer, 'group_size') and layer.group_size > 0

        if use_per_group:
            return torch_npu.npu_weight_quant_batchmatmul(
                x=x,
                weight=layer.weight,
                antiquant_scale=scale,
                antiquant_offset=offset,
                bias=bias,
                antiquant_group_size=layer.group_size,
            )
        else:
            return torch_npu.npu_weight_quant_batchmatmul(
                x=x,
                weight=layer.weight,
                antiquant_scale=scale,
                antiquant_offset=offset,
                bias=bias,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after model loading.

        msmodelslim format:
        - Weight: [output_size // 2, input_size] (packed int8 along output dim)
        - Scale: [output_size, num_groups]

        Processing:
        - Unpack int4 weights (2 per int8) along output dimension
        - Convert unsigned nibbles to signed int8
        - Transpose weights from [N, K] to [K, N] for NPU API
        - Transpose scales from [N, num_groups] to [num_groups, N] for NPU API
        - Store group_size in layer for use in apply()
        """
        # Determine if per-group quantization is used based on scale shape
        scale_shape = layer.weight_scale.data.shape
        is_per_group = len(scale_shape) == 2 and scale_shape[1] > 1

        if is_per_group:
            # Per-group quantization
            layer.group_size = self.group_size
        else:
            # Per-channel quantization
            layer.group_size = 0

        # Unpack int4 weights: each int8 contains 2 int4 values
        # Weight shape: [output_size // 2, input_size] -> [output_size, input_size]
        packed_weight = layer.weight.data  # [N//2, K]

        # Extract low and high nibbles (4-bit values as unsigned 0-15)
        low_nibble = (packed_weight & 0x0F).to(torch.int8)
        high_nibble = ((packed_weight >> 4) & 0x0F).to(torch.int8)

        # Convert from unsigned nibble representation to signed int8
        # The int4 values are stored with offset 8: 0-7 stay as-is, 8-15 map to -8 to -1
        low_nibble = torch.where(low_nibble >= 8, low_nibble - 16, low_nibble)
        high_nibble = torch.where(high_nibble >= 8, high_nibble - 16, high_nibble)

        # Interleave along output dimension to get [N, K]
        # Pattern: low, high, low, high, ... along dim 0
        output_size_packed = packed_weight.shape[0]
        input_size = packed_weight.shape[1]
        unpacked_weight = torch.empty(
            output_size_packed * 2, input_size, dtype=torch.int8, device=packed_weight.device
        )
        unpacked_weight[0::2, :] = low_nibble
        unpacked_weight[1::2, :] = high_nibble

        # Transpose from [N, K] to [K, N] for NPU API
        layer.weight.data = unpacked_weight.transpose(0, 1).contiguous()

        # Apply optional NZ format conversion
        layer.weight.data = maybe_trans_nz(layer.weight.data)

        # Process scales and offsets
        if is_per_group:
            # Per-group: transpose from [N, num_groups] to [num_groups, N]
            layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()
            layer.weight_offset.data = layer.weight_offset.data.transpose(0, 1).contiguous()
        else:
            # Per-channel: flatten to [N]
            layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
            layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
