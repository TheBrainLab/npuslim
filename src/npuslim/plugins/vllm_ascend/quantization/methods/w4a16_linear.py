"""W4A16 Linear quantization scheme for Ascend NPU.

Registered with vllm-ascend via a version-gated registrar wrapper around
@register_scheme. Aligned with NPUSlim's column-wise (input dimension)
packing format.
"""

import torch
import torch_npu
from typing import Any

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme
from vllm_ascend.utils import maybe_trans_nz

from npuslim.plugins.registry import package_version_range, register_patch


# Try to import upstream unpack_from_int32 utility
try:
    from vllm_ascend.quantization.methods.w4a16 import unpack_from_int32
    HAS_UPSTREAM_UNPACK = True
except ImportError:
    HAS_UPSTREAM_UNPACK = False

    def unpack_from_int32(
        weight: torch.Tensor,
        shape: torch.Size,
        num_bits: int,
        packed_dim: int = 1,
    ) -> torch.Tensor:
        """Fallback unpack implementation matching upstream signature.

        Args:
            weight: The packed int32 tensor containing quantized weights
            shape: Original shape to restore
            num_bits: The number of bits used for quantization (<= 8)
            packed_dim: Dimension along which weights are packed (0 or 1)

        Returns:
            Unpacked tensor with int8 dtype after applying offset correction
        """
        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1
        offset = pow(2, num_bits) // 2

        if packed_dim == 1:
            unpacked = torch.zeros(
                (weight.shape[0], weight.shape[1] * pack_factor),
                device=weight.device, dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
            unpacked = unpacked[:, :shape[1]]
        else:  # packed_dim == 0
            unpacked = torch.zeros(
                (weight.shape[0] * pack_factor, weight.shape[1]),
                device=weight.device, dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
            unpacked = unpacked[:shape[0], :]

        return (unpacked - offset).to(torch.int8)


@register_patch(
    registrar=register_scheme("W4A16", "linear"),
    condition=package_version_range("vllm_ascend", min_version="0.1.0"),
)
class AscendW4A16LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A16.

    This scheme uses 4-bit quantized weights with 16-bit activations.
    Supports both per-channel and per-group quantization with int32 packed
    weights (8 int4 per int32), aligned with upstream vllm-ascend format.

    NPUSlim format (column-wise packing):
    - Weight: [output_size, input_size // 8] (packed int4 along input dim)
    - Scale: [output_size, num_groups] where num_groups = input_size // group_size
    - Offset: [output_size, num_groups]

    After processing:
    - Weight: [input_size, output_size] (transposed for NPU API)
    - Scale: [num_groups, output_size] (transposed for NPU API)
    """

    def __init__(self) -> None:
        self.pack_factor = 8  # 8 int4 per int32
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

        For W4A16, weights are stored as int32 (packed int4).
        Format: [output_size, input_size // 8] (input packed, column-wise)
        """
        assert input_size % self.pack_factor == 0, (
            f"Expecting `input_size` {input_size} "
            f"can be divided by `pack_factor` {self.pack_factor}"
        )

        params_dict: dict[str, Any] = {}
        # Weight shape: [output_size, input_size // 8] as int32
        # Each int32 contains 8 int4 values packed along input dimension
        params_dict["weight"] = torch.empty(
            output_size, input_size // self.pack_factor, dtype=torch.int32
        )
        # Tell vLLM's weight_loader about the packed dimension
        # packed_dim=1 means input dimension (dim 1) is packed
        # packed_factor=8 means each element represents 8 packed values
        params_dict["_packed_dim"] = 1
        params_dict["_packed_factor"] = self.pack_factor
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

        NPUSlim int32 format (column-wise packing):
        - Weight: [output_size, input_size // 8] (packed int4 along input dim)
        - Scale: [output_size, num_groups]

        Processing:
        - Unpack int4 weights using unpack_from_int32() utility
        - Transpose weights from [N, K] to [K, N] for NPU API
        - Transpose scales from [N, num_groups] to [num_groups, N] for NPU API
        - Store group_size in layer for use in apply()
        """
        # Determine quantization type from scale shape
        scale_shape = layer.weight_scale.data.shape
        is_per_group = len(scale_shape) == 2 and scale_shape[1] > 1

        if is_per_group:
            layer.group_size = self.group_size
        else:
            layer.group_size = 0

        # Unpack int4 weights from int32 using utility
        packed_weight = layer.weight.data  # [N, K//8] for column-wise packing
        output_size, packed_input = packed_weight.shape
        input_size = packed_input * self.pack_factor

        # Use unpack_from_int32 with packed_dim=1 (input dimension packed)
        unpacked_weight = unpack_from_int32(
            weight=packed_weight,
            shape=torch.Size([output_size, input_size]),
            num_bits=4,
            packed_dim=1,
        )
        # unpacked_weight is now int8, shape [output_size, input_size]

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
