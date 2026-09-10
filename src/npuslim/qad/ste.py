"""STE fake-quant：forward 复刻部署量化网格，backward 恒等直通。

复刻语义来源：src/npuslim/algorithms/quantization/gptq/gptq_algo.py::_pack_ascend
- 对称 per-output-channel：s = amax / (2^(bits-1) - 1)（zero-point = 0）
- int4：q = clamp(round(w/s), -8, 7)，有效权重 = q*s（补码网格）
- int8：q = clamp(round(w/s), -128, 127)

注意：部署链路中 scale_bias = 8·Σ(q·s) 是 kernel 对激活 int8 表征的无条件补偿项，
在 float 训练路径中不存在（float 激活直接与有效权重 q*s 相乘），故 STE 不含 scale_bias。
"""
from __future__ import annotations

import torch


def compute_weight_scale(w: torch.Tensor, bits: int) -> torch.Tensor:
    """对称 per-output-channel scale [out, 1]，与 GPTQ 打包路径的网格定义一致。"""
    qmax = 2 ** (bits - 1) - 1  # int4 -> 7, int8 -> 127
    amax = w.abs().amax(dim=-1, keepdim=True).float()
    return (amax / qmax).clamp_min(1e-12)


class _STEQuant(torch.autograd.Function):
    """forward: clamp(round(w/s))·s；backward: 恒等直通。"""

    @staticmethod
    def forward(ctx, w: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        q = torch.clamp(torch.round(w / scale), qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out, None, None, None


def fake_quant_weight(w: torch.Tensor, bits: int = 4, scale: torch.Tensor | None = None) -> torch.Tensor:
    """权重 fake-quant（float 路径的有效权重）。

    Args:
        w: [out, in] float 权重（已含 AWQ 折叠后的最终形态）
        bits: 4（W4A8 专家）或 8（W8A8 稠密）
        scale: 可选外部 scale [out,1]（如从 PTQ 产物冻结；None 则按当前 w 重算）
    """
    s = scale if scale is not None else compute_weight_scale(w, bits)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return _STEQuant.apply(w, s.to(w.dtype), qmin, qmax)


def effective_weight(w: torch.Tensor, delta: torch.Tensor | None, bits: int = 4) -> torch.Tensor:
    """QAD 有效权重：fakequant(W) + delta（output-space LoRA，delta 不重量化）。

    与 InstinctRazor --fast 模式及 Phase D（merge 后重跑 PTQ）的部署一致性对齐：
    LoRA 输出以 float 叠加在量化有效权重之后。
    """
    wq = fake_quant_weight(w, bits)
    return wq if delta is None else wq + delta
