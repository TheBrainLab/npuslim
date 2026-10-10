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
    """forward: clamp(round(w/s))·s；backward: w 恒等直通 + scale 取 LSQ 式梯度。

    scale 梯度（LSQ 约定）：∂(v·s)/∂s ≈ v（round/clamp 输出视为常数），
    即 dL/ds = Σ grad_out·v——支撑 §5.14-B1 scale-only/可训练 scale 对照臂。
    scale 为普通张量（不可训练）时 grad_s 弃用，行为与历史版本一致。
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int):
        v = w / scale
        v = v.round_()            # 原地 round（省 fp32 临时，数值同 clamp(round(w/s))）
        v = v.clamp_(qmin, qmax)  # 原地 clamp
        # 条件保存（0925 修复）：v 仅供 scale 梯度（LSQ）使用——scale 冻结时
        # backward 不读 v。无条件保存会让 deploy 复合（fakequant 在图内执行、
        # no_grad 屏蔽不了）每层每步多存一个与 w 同形状的 fp32 v
        # （gate_up ~1.6GB + down ~0.4GB × 150 次/步）→ 大块 alloc/free 的
        # 池碎片累积 → bwd 逐步退化（E2 实测 33s→62-166s，phase2b/post 因
        # no_grad 不建图而平坦）。语义不变：可训练 scale 路径照常保存。
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(v)
            return v * scale
        # scale 冻结：v 未被保存 → 原地反量化（0925 修复 3a）。数值同 v*scale（同一
        # broadcast 乘法）；省 1 个与 w 同形状的满张量/调用——deploy 复合每次训练步
        # 经 checkpoint 前向重放 + 反向重算各跑一遍（~300 次/步，gate_up slab fp32
        # 800MB / down 400MB），此前 v*scale 是 bwd 重算路径 allocator 流量的主力之一。
        return v.mul_(scale)

    @staticmethod
    def backward(ctx, grad_out):
        grad_w = grad_out if ctx.needs_input_grad[0] else None
        grad_s = None
        if ctx.needs_input_grad[1]:
            (v,) = ctx.saved_tensors
            grad_s = (grad_out * v).sum(dim=-1, keepdim=True)
        return grad_w, grad_s, None, None


def fake_quant_weight(w: torch.Tensor, bits: int = 4, scale: torch.Tensor | None = None) -> torch.Tensor:
    """权重 fake-quant（float 路径的有效权重）。

    Args:
        w: [out, in] float 权重（已含 AWQ 折叠后的最终形态）
        bits: 4（W4A8 专家）或 8（W8A8 稠密）
        scale: 可选外部 scale [out,1]（如从 PTQ 产物冻结；None 则按当前 w 重算）。
               可为 requires_grad 的叶张量（可训练 scale，任务 2）——梯度经 LSQ 式
               直通回传。
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


def activation_dynamic_int8(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """per-token 动态对称 int 激活 fake-quant（部署语义，W4A8 的 A8 侧；单一语义源）。

    定义（与部署 kernel 的 per-token 动态量化一致）：
        s_t = max_i |x_{t,i}| / qmax          （逐 token；qmax = 2^(bits-1) - 1）
        q   = clamp(round(x / s_t), -qmax-1, qmax)
        x̂   = q · s_t

    训练图内若需模拟激活量化、以及评测 hook（eval_rotated_ppl 的同名函数为
    其别名）都必须复用本函数，避免训练/评测/部署三方语义漂移。
    """
    qmax = 2 ** (bits - 1) - 1
    s = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return (x.float() / s).round().clamp(-qmax - 1, qmax).mul(s).to(x.dtype)
