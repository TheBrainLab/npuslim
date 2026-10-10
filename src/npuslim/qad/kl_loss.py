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


def _per_row_fwd_rev(student_logits: torch.Tensor, teacher_topk_ids: torch.Tensor,
                     teacher_topk_logprobs: torch.Tensor, vocab_size: int,
                     tau: float, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """逐行 forward/reverse KL（top-K + 均匀尾近似）。返回 (fwd[B], rev[B])。"""
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

    fwd = (p * (log_p - s_lp)).sum(dim=-1) + rest * (log_rest - log_q_rest)
    rev = (q * (s_lp - log_p)).sum(dim=-1) + q_rest * (log_q_rest - log_rest)
    return fwd, rev


def _teacher_label_conf(teacher_topk_ids: torch.Tensor, teacher_topk_logprobs: torch.Tensor,
                        label_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """p_teacher(label)：label ∈ top-K → 其概率 exp(log_p)；否则均匀尾 rest/(V-K)。"""
    log_p = teacher_topk_logprobs.float()
    p = torch.exp(log_p)
    rest = (1.0 - p.sum(dim=-1)).clamp_min(0.0)
    v_rest = max(vocab_size - teacher_topk_ids.shape[-1], 1)
    tail_conf = rest / v_rest
    match = teacher_topk_ids == label_ids.unsqueeze(-1)        # [B, K]
    in_topk = match.any(dim=-1)                                # [B]
    pos = match.float().argmax(dim=-1)                         # 首个匹配位（未匹配行为 0，弃用）
    conf_topk = torch.gather(p, dim=-1, index=pos.unsqueeze(-1)).squeeze(-1)
    return torch.where(in_topk, conf_topk, tail_conf)


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
    fwd, rev = _per_row_fwd_rev(student_logits, teacher_topk_ids, teacher_topk_logprobs,
                                vocab_size, tau, chunk)
    return (fwd if mode == "forward" else rev).mean()


def cakld_distill_loss(
    student_logits: torch.Tensor,          # [B, V]
    teacher_topk_ids: torch.Tensor,        # [B, K] int64
    teacher_topk_logprobs: torch.Tensor,   # [B, K] float（log_softmax 截断值）
    label_ids: torch.Tensor,               # [B] 真实 next token（cache 列 0）
    vocab_size: int,
    tau: float = 1.0,
    chunk: int = 32768,
) -> torch.Tensor:
    """CAKLD（Contextual Asymmetric KL，AngelSlim loss.py cakld 语义）：
    按 teacher 对 label 的置信度逐 token 混合 forward/reverse KL。

    conf_t = p_teacher(label_t)：
      label_t ∈ teacher top-K → 其概率 exp(log_p)（精确）
      label_t ∈ 尾部        → 均匀尾近似 rest/(V-K)
    loss_t = conf_t·rev_kl_t + (1−conf_t)·fwd_kl_t
    （teacher 置信 → reverse 偏 mode-seeking；不确定 → forward 偏 mode-covering）
    """
    fwd, rev = _per_row_fwd_rev(student_logits, teacher_topk_ids, teacher_topk_logprobs,
                                vocab_size, tau, chunk)
    conf = _teacher_label_conf(teacher_topk_ids, teacher_topk_logprobs, label_ids, vocab_size)
    return (conf * rev + (1.0 - conf) * fwd).mean()
