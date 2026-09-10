"""Chunked KL 蒸馏 loss（QAD / on-policy OPD 共用）。

数学（逐项对照 InstinctRazor opd_train_fsdp.py L263-286，2026-09-02 复核）：
- teacher 只提供 top-K logprobs（log_softmax 截断），top-K 外的概率质量
  rest = 1 - Σ exp(p_t) 均摊到 (V-K) 个 token：log_rest = log(rest) - log(V-K)
  （质量守恒，防止 top-K 外被当作零概率）
- forward: Σ_topk p·(log p - s_lp) + rest·(log_rest - log q_rest)
  reverse: Σ_topk q·(s_lp - log p) + q_rest·(log q_rest - log_rest)
- 分块策略差异：InstinctRazor 沿位置维 T 分块（VCHUNK=128，每块全词表
  logsumexp）；本实现沿词表维 V 分块在线 logsumexp（fp32，峰值 O(B·chunk)）——
  数学等价，均避免全量 [B,V] fp32 softmax 中间张量
- 已知特性：缺口趋零时 loss 可轻微转负（partial top-K KL + 均匀尾近似），
  与参考实现一致；作监控指标看相对下降即可

npz 缓存格式（§12.9）：token_ids[L, K+1]（列 0 = 真实 token，列 1..K = top-K）
与 logits[L, K+1]（log_softmax 值，列 0 = 真实 token 的 log 概率）。
"""
from __future__ import annotations

import torch


def _chunked_logsumexp(z: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """[B, V] 沿 V 在线分块 logsumexp，fp32、峰值内存 O(B·chunk)。"""
    z = z.float()
    b, v = z.shape
    m = z.new_full((b,), float("-inf"))
    s = z.new_zeros((b,))
    for lo in range(0, v, chunk):
        c = z[:, lo:lo + chunk]
        cm = c.max(dim=-1).values
        m_new = torch.maximum(m, cm)
        s = s * torch.exp(m - m_new) + torch.exp(c - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new
    return m + torch.log(s.clamp_min(1e-38))


def kl_distill_loss(
    student_logits: torch.Tensor,          # [B, V] 未经 softmax 的输出（已除 tau 由调用方或此处处理）
    teacher_topk_ids: torch.Tensor,        # [B, K] int64
    teacher_topk_logprobs: torch.Tensor,   # [B, K] float（log_softmax 截断值）
    vocab_size: int,
    mode: str = "forward",                 # forward | reverse
    tau: float = 1.0,
    chunk: int = 32768,
) -> torch.Tensor:
    """top-K 截断 KL 蒸馏 loss（标量）。

    forward KL = Σ_t p_t·(log p_t - s_lp) + rest·(log_rest - log q_rest)
    reverse KL = Σ_t q_t·(s_lp - log p_t) + q_rest·(log q_rest - log_rest)
    其中 s_lp = student 在 teacher top-K 位置上的 logprob（/tau 后），
    q_rest = 1 - Σ_t exp(s_lp)，log_rest = log(rest) - log(V-K)。
    """
    assert mode in ("forward", "reverse"), mode
    student_logits = student_logits.float()
    b, v = student_logits.shape
    k = teacher_topk_ids.shape[-1]
    assert v >= k and vocab_size >= v

    z = student_logits / tau
    lse = _chunked_logsumexp(z, chunk)                       # [B]
    s_lp = z.gather(-1, teacher_topk_ids) - lse.unsqueeze(-1)  # [B, K] student logprob @ teacher ids

    # teacher 侧
    log_p = teacher_topk_logprobs.float()                     # log_softmax 截断
    p = torch.exp(log_p)
    rest = (1.0 - p.sum(dim=-1)).clamp_min(0.0)               # [B] top-K 外质量
    log_rest = torch.log(rest.clamp_min(1e-12)) - torch.log(
        torch.tensor(float(vocab_size - k), device=z.device)
    )

    q = torch.exp(s_lp)
    q_sum = q.sum(dim=-1).clamp_max(1.0 - 1e-7)
    q_rest = 1.0 - q_sum
    log_q_rest = torch.log(q_rest.clamp_min(1e-12))

    if mode == "forward":
        kl_topk = (p * (log_p - s_lp)).sum(dim=-1)
        kl_rest = rest * (log_rest - log_q_rest)
    else:
        kl_topk = (q * (s_lp - log_p)).sum(dim=-1)
        kl_rest = q_rest * (log_q_rest - log_rest)
    return (kl_topk + kl_rest).mean()
