"""Sparse24 2:4 structured sparse linear scheme for Ascend NPU.

Registered with vllm-ascend via @register_scheme("Sparse24", "linear").
Loads densified sparse weights + tiled index from NPUSlim SparseGPT output
and uses the AscendC sparse_matmul_4to2 kernel for inference.
"""

import math
from typing import Any

import torch

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme

from npuslim.ops.sparse_matmul import sparse_matmul_4to2

_K_TILE_INDEX = 8


@register_scheme("Sparse24", "linear")
class AscendSparse24LinearMethod(AscendLinearScheme):
    """Linear scheme for 2:4 structured sparse weights on Ascend NPU.

    Weight format (from NPUSlim SparseGPT):
    - weight:       [output_size, input_size // 2] int8  (densified non-zeros)
    - weight_scale: [output_size] float16               (per-channel symmetric)
    - weight_index: 1D uint8                            (tiled index for AscendC)
    """

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert input_size % 4 == 0, (
            f"Sparse24 requires input_size divisible by 4, got {input_size}"
        )
        self._input_size = input_size
        self._output_size = output_size

        pad_n = math.ceil(output_size / 16) * 16
        k8 = input_size // 8
        num_tiles = math.ceil(k8 / _K_TILE_INDEX)
        index_size = num_tiles * pad_n * _K_TILE_INDEX

        return {
            "weight": torch.empty(
                output_size, input_size // 2, dtype=torch.int8
            ),
            "_packed_dim": 1,
            "_packed_factor": 2,
        }

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        input_size = self._input_size

        pad_n = math.ceil(output_size / 16) * 16
        k8 = input_size // 8
        num_tiles = math.ceil(k8 / _K_TILE_INDEX)
        index_size = num_tiles * pad_n * _K_TILE_INDEX

        return {
            "weight_scale": torch.empty(output_size, dtype=torch.float16),
            "weight_index": torch.zeros(index_size, dtype=torch.uint8),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])

        max_val = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_scale = max_val / 127.0
        x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

        c_int32 = sparse_matmul_4to2(x_int8, layer.weight, layer.weight_index)

        out = c_int32.float() * (x_scale * layer.weight_scale.unsqueeze(0))
        out = out.to(x.dtype).reshape(*orig_shape, layer.weight.shape[0])

        if bias is not None:
            out = out + bias
        return out
