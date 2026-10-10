"""SSZ (Scan-Scale-Zero) per-channel scale refinement for weight quantization.

Ported from msmodelslim `core/quantizer/impl/ssz.py` (Mulan PSL v2, MindStudio
official reference). The algorithm iteratively refines the per-output-channel
scale to minimize the weight quantization MSE:

  1. start from an initial scale (e.g. minmax);
  2. quantize, then compute the least-squares optimal scale for the current
     quantized values: s = sum(w*q) / sum(q^2);
  3. re-quantize with the new scale, compare per-channel MSE, keep the best
     scale greedily;
  4. repeat until convergence or `step` iterations.

This module implements the **symmetric** branch only (offset == 0 in the signed
quant grid), which matches the npuslim W4A8/W8A8 per-channel symmetric pack
protocol (q = clamp(round(w/s) + zero, 0, maxq), zero = (maxq+1)/2, dequant =
s*(q - zero)). It is a pure-torch, vectorized variant that operates on an
arbitrary number of rows ([N, in]) at once, so it works both per-expert and
across a batched expert chunk.

Reference file:
    /data/yult/llm_inference/msmodelslim/msmodelslim/core/quantizer/impl/ssz.py
"""

from __future__ import annotations

import torch
from loguru import logger

SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10
SCALE_SEARCH_MIN_SCALE = 1e-30
SCALE_SEARCH_DEFAULT_STEP = 10


@torch.no_grad()
def ssz_calculate_scale(
    w: torch.Tensor,
    initial_scale: torch.Tensor,
    bits: int = 4,
    step: int = SCALE_SEARCH_DEFAULT_STEP,
    converge_threshold: float = SCALE_SEARCH_CONVERGE_THRESHOLD,
    min_scale: float = SCALE_SEARCH_MIN_SCALE,
) -> torch.Tensor:
    """Iteratively refine per-row scale via least-squares (symmetric SSZ).

    Args:
        w: [N, in] float weight matrix (N rows = output channels to refine)
        initial_scale: [N, 1] initial per-row scale (e.g. minmax)
        bits: quantization bit width (4 or 8)
        step: max SSZ iterations (modelslim production yaml uses 10)
        converge_threshold: relative/absolute MSE improvement threshold
        min_scale: clamp floor to avoid division by zero

    Returns:
        [N, 1] refined per-row scale
    """
    w = w.float()
    maxq = 2 ** int(bits) - 1
    zero = (maxq + 1) / 2  # symmetric mid offset (matches compute_scales_with_zero)

    s = initial_scale.float().clamp_min(min_scale)
    q = torch.clamp(torch.round(w / s) + zero, 0, maxq)
    best_scale = s.clone()
    best_mse = torch.mean(
        torch.pow(torch.abs(w - s * (q - zero)), 2), dim=1, keepdim=True
    )

    for _ in range(int(step)):
        v = q - zero  # signed quantized values [N, in]
        # least-squares optimal scale for the current quantized values
        current_scale = (torch.sum(w * v, dim=1, keepdim=True) /
                         torch.sum(v * v, dim=1, keepdim=True).clamp(min=min_scale))
        current_scale = current_scale.clamp_min(min_scale)
        s = current_scale
        q = torch.clamp(torch.round(w / s) + zero, 0, maxq)
        current_mse = torch.mean(
            torch.pow(torch.abs(w - s * (q - zero)), 2), dim=1, keepdim=True
        )

        # convergence: every channel satisfies a relative or absolute threshold
        mask1 = (best_mse - current_mse) / best_mse.clamp(min=1e-4) < converge_threshold
        mask2 = torch.abs(best_mse - current_mse) < converge_threshold
        if (torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))).sum() == 0:
            break

        # greedy per-channel keep-best
        mask = (current_mse < best_mse).to(torch.int32)
        best_mse = best_mse * (1 - mask) + current_mse * mask
        best_scale = best_scale * (1 - mask) + current_scale * mask

    return best_scale


def clip_search_scale(
    w: torch.Tensor,
    initial_scale: torch.Tensor,
    bits: int = 4,
    lo: float = 0.65,
    hi: float = 1.0,
    steps: int = 20,
    with_ssz: bool = False,
    ssz_step: int = 10,
) -> torch.Tensor:
    """OmniQuant / AWQ-clip 家族：在 clip 比例网格上选 per-row MSE 最优 scale。

    w: [rows, in] 权重；initial_scale: [rows, 1]（minmax 或 SSZ 初始 scale）。
    对 a ∈ linspace(hi, lo, steps)：s_a = a * initial_scale；
      with_ssz=True 时再经 SSZ 精修（= 探针 clip_then_ssz）；
    逐行取 MSE 最小的 s_a。返回 [rows, 1]。
    参考：workspace/glm5.2_deploy/p0_quant_method_probe.py（per-channel clip-search
    权重 MSE 比纯 SSZ 低 23.5%，clip+SSZ 低 26.2%）。
    """
    qmax = 2 ** (bits - 1) - 1
    best_scale = initial_scale.clone()
    best_err = None
    for a in torch.linspace(hi, lo, steps).tolist():
        s = initial_scale * a
        if with_ssz:
            s = ssz_calculate_scale(w, s, bits=bits, step=ssz_step)
        q = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
        err = (q - w).pow(2).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, s
        else:
            better = err < best_err
            best_scale = torch.where(better, s, best_scale)
            best_err = torch.where(better, err, best_err)
    return best_scale


def omni_refine_scale(
    w: torch.Tensor,
    initial_scale: torch.Tensor,
    x_cm: torch.Tensor,
    bits: int = 4,
    steps: int = 60,
    lr: float = 0.02,
    u_range: float = 1.0,
) -> torch.Tensor:
    """OmniQuant 式 per-row scale 梯度精修（LWC 思想，专家行级；§6.8.3 中间路径 / §6.9.2#3）。

    目标 = 「真实量化器前向 MSE」的期望形式（§6.5 定义）：

        L(s) = mean_{r,j} ( ΔW @ Xᵀ )²,   ΔW = W − Q(W; s)

    其中 Q 为真量化器（symmetric int4/int8 网格；对 s 的梯度取 LSQ 式直通：
    round/clamp 输出 v 视为常数，dL/ds = 2(s·v−w)·v，饱和元素方向正确），
    x_cm 为该专家捕获的校准激活 [in, n]（channel-major；n 为路由到该专家的
    token 数，n << in 时此低秩形式远快于稠密 H = X Xᵀ 的 matmul）。
    与 clip_search_scale 的差异：clip+SSZ 是权重-only MSE 上的网格/LSQ 搜索；
    本函数以激活加权目标对同一 scale 做 warm-start 梯度精修（best-loss 保留，
    保证精修不劣于初值），参数化 log s（|Δlog s| ≤ u_range），manual Adam
    全批量、确定性。X 按全局 abs-max 归一（loss 均匀缩放，argmin 不变，
    同时保证 fp16 matmul 不溢出）。

    Args:
        w: [rows, in] float 权重（rows = CH*out 展平，行序 expert-major）
        initial_scale: [rows, 1] warm-start scale（clip+SSZ 输出）
        x_cm: [in, n] channel-major 校准激活（fp16 CPU 块 cat 后搬运）
        bits: 量化位宽（4 或 8）
        steps: Adam 步数（0 = 关闭，原样返回）
        lr: Adam 学习率（作用于 log s）
        u_range: log s 相对初值的漂移钳制（±e^u_range）

    Returns:
        [rows, 1] 精修 scale（device 与 w 一致，float32）
    """
    if steps <= 0 or x_cm.numel() == 0:
        return initial_scale

    w = w.float()
    lo_q = -(2 ** (bits - 1))
    hi_q = 2 ** (bits - 1) - 1
    s0 = initial_scale.float().clamp_min(1e-12)
    u0 = s0.log()

    # NPU 上 bf16 走 tensor core 且指数域与 fp32 相同——up-down AWQ 折叠后 down_proj
    # 权重幅度 ~1e-2，fp16 matmul 输出分辨率会淹没真改进（实测 rel_impr 恒 0）。
    # 损失按 w/x 的 RMS 归一（两侧常数因子，argmin 不变），使被优化量保持 O(1)。
    matmul_dtype = torch.bfloat16 if w.device.type == "npu" else torch.float32
    x_rms = x_cm.pow(2).mean().sqrt().clamp_min(1e-12)
    w_rms = w.abs().mean().clamp_min(1e-12)
    x_t = (x_cm / x_rms).to(device=w.device, dtype=matmul_dtype)

    def _loss(u_val: torch.Tensor) -> torch.Tensor:
        s = u_val.exp()
        v = torch.clamp(torch.round(w / s), min=lo_q, max=hi_q)
        dw = ((s * v.detach() - w) / w_rms).to(matmul_dtype)
        y = dw @ x_t
        return y.float().pow(2).mean()

    u = u0.detach().clone().requires_grad_(True)
    m = torch.zeros_like(u)
    v = torch.zeros_like(u)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_u = u0.detach().clone()
    best_loss: torch.Tensor | None = None
    first_loss: float | None = None
    with torch.enable_grad():
        for step_i in range(1, int(steps) + 1):
            loss = _loss(u)
            (grad,) = torch.autograd.grad(loss, u)
            with torch.no_grad():
                if best_loss is None:
                    best_loss = loss.detach().clone()
                    first_loss = float(loss.item())
                else:
                    better = loss.detach() < best_loss
                    best_loss = torch.where(better, loss.detach(), best_loss)
                    best_u = torch.where(better, u.detach(), best_u)
                # manual Adam（避免 torch.optim 在 torch_npu 上的内核兼容风险）
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                u -= lr * (m / (1 - beta1 ** step_i)) / (
                    (v / (1 - beta2 ** step_i)).sqrt() + eps
                )
                u.clamp_(u0 - u_range, u0 + u_range)

    assert best_loss is not None and first_loss is not None
    rel = (first_loss - float(best_loss)) / first_loss if first_loss > 0 else 0.0
    logger.info(
        f"[SSZ] omni refine: rows={w.shape[0]} n={x_cm.shape[1]} steps={steps} "
        f"rel_impr={rel:.4f} ({first_loss:.3e} -> {float(best_loss):.3e})"
    )
    return best_u.exp()
