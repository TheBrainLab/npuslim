import torch
import torch.nn as nn
import numpy as np
import math

from npuslim.utils.backend import bh


class GPTQQuantLinear(nn.Module):

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        # Backend-aware buffer registration
        self._is_ascend_format = bh.name == "npu"

        if self._is_ascend_format:
            # Ascend NPU backend only supports 4-bit quantization
            assert bits == 4, f"Ascend backend only supports 4-bit, got {bits}"

            # Ascend-specific pack factor: 8 int4 values per int32
            pack_factor = 8
            assert outfeatures % pack_factor == 0, (
                f"Ascend backend requires outfeatures {outfeatures} divisible by {pack_factor}"
            )

            # weight: [out//8, in] packed int4 into int32; scale/offset: [out, groups] bfloat16
            self.register_buffer(
                "weight",
                torch.zeros((outfeatures // pack_factor, infeatures), dtype=torch.int32),
            )
            num_groups = math.ceil(infeatures / self.group_size)
            self.register_buffer(
                "weight_scale",
                torch.zeros((outfeatures, num_groups), dtype=torch.bfloat16),
            )
            self.register_buffer(
                "weight_offset",
                torch.zeros((outfeatures, num_groups), dtype=torch.bfloat16),
            )
            # No g_idx needed for Ascend format as grouping is handled during packing

        else:
            # GPTQ format: packed int32 weights (original behavior)
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (infeatures // 32 * self.bits, outfeatures), dtype=torch.int32
                ),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (
                        math.ceil(infeatures / self.group_size),
                        outfeatures // 32 * self.bits,
                    ),
                    dtype=torch.int32,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(infeatures / self.group_size), outfeatures),
                    dtype=weight_dtype,
                ),
            )
            self.register_buffer(
                "g_idx",
                torch.tensor(
                    [i // self.group_size for i in range(infeatures)], dtype=torch.int32
                ),
            )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

        # Unpacking factors for GPTQ format (only used for GPU/CPU)
        if not self._is_ascend_format:
            if self.bits in [2, 4, 8]:
                self.wf = torch.tensor(
                    list(range(0, 32, self.bits)), dtype=torch.int32
                ).unsqueeze(0)
            elif self.bits == 3:
                self.wf = torch.tensor(
                    [
                        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                        [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                        [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                    ],
                    dtype=torch.int32,
                ).reshape(1, 3, 12)

    def post_init(self):
        pass

    # =========================================================================
    # Ascend NPU Backend: Pack and Forward
    # =========================================================================

    def _pack_ascend(self, linear, scales, zeros, g_idx=None):
        """Pack weights in Ascend NPU int32 format.

        vLLM-Ascend expects:
          weight:       [outfeatures // 8, infeatures] as int32 (packed int4)
          weight_scale: [outfeatures, num_groups] as bfloat16
          weight_offset: [outfeatures, num_groups] as bfloat16

        Packing pattern: 8 int4 values per int32, packed along output dimension.
        int32[i] = int4[8*i] | (int4[8*i+1] << 4) | ... | (int4[8*i+7] << 28)
        """
        W = linear.weight.data.clone()
        outfeatures, infeatures = W.shape
        device = W.device
        pack_factor = 8  # 8 int4 values per int32

        # Generate group indices if not provided
        if g_idx is None:
            g_idx = torch.arange(infeatures, device=device) // self.group_size

        # Vectorized quantization: broadcast per-group scales/zeros to per-column
        # scales[:, g_idx] -> (outfeatures, infeatures) where col i uses scales[:, g_idx[i]]
        current_scales = scales[:, g_idx]
        current_scale_zeros = (zeros * scales)[:, g_idx]

        # Quantize to signed int4 range [-8, 7]
        signed_offset = 2 ** (self.bits - 1)  # 8 for 4-bit
        q = torch.round((W + current_scale_zeros) / current_scales) - signed_offset
        intweight_2d = q.clamp(-signed_offset, signed_offset - 1).to(torch.int8)

        # Convert signed int4 [-8, 7] to unsigned [0, 15] for packing
        intweight_unsigned = (intweight_2d + signed_offset).to(torch.uint8)

        # Pack 8 int4 values into int32 along output dimension
        # Shape: [outfeatures, infeatures] -> [outfeatures // 8, infeatures]
        packed_out = outfeatures // pack_factor
        packed_weight = torch.zeros((packed_out, infeatures), dtype=torch.int32, device=device)

        for i in range(pack_factor):
            # Each group of 8 consecutive rows packs into one int32
            # Row 8*i+k goes to packed[i] at bits [4*k, 4*k+3]
            row_indices = torch.arange(i, outfeatures, pack_factor, device=device)
            packed_weight |= (intweight_unsigned[row_indices, :].to(torch.int32) << (self.bits * i))

        self.weight = packed_weight.contiguous()
        self.weight_scale = scales.to(torch.bfloat16)

        # Set weight_offset=0 as the signed shift is already in packed weights
        self.weight_offset = torch.zeros_like(zeros, dtype=torch.bfloat16)

        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

    def _forward_ascend(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Ascend NPU int32 format.

        Unpacks packed int4 weights from int32 and performs dequantized matmul.
        Used for calibration verification; vLLM-Ascend handles unpacking at runtime.
        """
        assert self.bits == 4, f"Ascend backend only supports 4-bit, got {self.bits}"

        # Packed weight shape: (outfeatures // 8, infeatures) int32
        # Scale/offset shape: (outfeatures, num_groups) bfloat16
        packed_weight = self.weight
        pack_factor = 8
        packed_out, in_feat = packed_weight.shape
        out_full = packed_out * pack_factor

        # Unpack int4 weights from int32: each int32 contains 8 int4 values
        # Pattern: packed[i] = int4[0] | (int4[1] << 4) | ... | (int4[7] << 28)
        num_bits = self.bits  # 4
        mask = (1 << num_bits) - 1  # 0xF
        offset = pow(2, num_bits) // 2  # 8 for signed conversion

        unpacked_weight = torch.zeros(
            out_full, in_feat, dtype=torch.int8, device=packed_weight.device
        )
        for i in range(pack_factor):
            # Extract 4-bit value and convert from unsigned [0,15] to signed [-8,7]
            unpacked_weight[i::pack_factor, :] = (
                ((packed_weight >> (num_bits * i)) & mask).to(torch.int8) - offset
            )

        # Dequantize using scale/offset: weight_fp = (int_weight + offset) * scale
        num_groups = self.weight_scale.shape[1]
        group_idx = torch.arange(self.infeatures, device=x.device) // self.group_size
        group_idx = group_idx.clamp(0, num_groups - 1)

        scale_per_col = self.weight_scale[:, group_idx]
        offset_per_col = self.weight_offset[:, group_idx]

        weight_fp = (
            unpacked_weight.to(torch.float32) + offset_per_col.float()
        ) * scale_per_col.float()

        return torch.matmul(x, weight_fp.T)

    # =========================================================================
    # GPTQ GPU Backend: Pack and Forward
    # =========================================================================

    def _pack_gptq(self, linear, scales, zeros, g_idx=None):
        """Pack weights in standard GPTQ format (int32 packed weights)."""
        W = linear.weight.data.clone()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (W[:, idx] + scale_zeros[self.g_idx[idx]])
                    / self.scales[self.g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]),
            dtype=np.uint32,
        )
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def _forward_gptq(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for standard GPTQ format (int32 packed weights)."""
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                self.wf.unsqueeze(0),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        elif self.bits == 3:
            zeros = self.qzeros.reshape(
                self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
            ).expand(-1, -1, -1, 12)
            zeros = zeros >> self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
                (zeros[:, :, 1, 0] << 2) & 0x4
            )
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
                (zeros[:, :, 2, 0] << 1) & 0x6
            )
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            )

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(
                self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
            )
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (
                weight - zeros[self.g_idx.long()]
            )
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(
                    scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()])
                )
            weights = torch.cat(weights, dim=1)

        return torch.matmul(x, weights)

    # =========================================================================
    # Public API: Dispatch to backend-specific implementations
    # =========================================================================

    def pack(self, linear, scales, zeros, g_idx=None):
        """Pack weights in backend-specific format."""
        if self._is_ascend_format:
            self._pack_ascend(linear, scales, zeros, g_idx)
        else:
            self._pack_gptq(linear, scales, zeros, g_idx)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        if self._is_ascend_format:
            out = self._forward_ascend(x)
        else:
            out = self._forward_gptq(x)

        out = out.to(x_dtype).reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out
