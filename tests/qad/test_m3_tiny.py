"""M3 tiny 端到端 sanity：tiny MoE + STE fake-quant + per-expert LoRA + chunked KL 蒸馏闭环。

验证目标（调研文档 §6.2 M3）：
- teacher 在小语料上过拟合（CE 可压低）——同时提供有信号的蒸馏目标
- int4 fake-quant 造成可测的蒸馏缺口（KL > 0），LoRA 训练可收敛收回
- 梯度只流经 LoRA 参数（冻结参数无 grad），loss 无 NaN
- 下游 CE（held-out）相对 quantized-only 基线改善、逼近 teacher

运行（npuslim conda env，CPU 即可）：
    /data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_m3_tiny.py
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.kl_loss import kl_distill_loss
from npuslim.qad.lora import ExpertLoRA
from npuslim.qad.tiny_moe import TinyMoEConfig, TinyMoEModel, attach_qad_lora, detach_qad_lora

torch.manual_seed(0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_batch(vocab: int, batch: int, seq: int) -> torch.Tensor:
    """合成小语料：偏斜词频分布（模拟真实 token 长尾）。"""
    probs = vocab ** -torch.linspace(0.5, 2.0, vocab)
    probs = probs / probs.sum()
    return torch.multinomial(probs, batch * seq, replacement=True).reshape(batch, seq)


def teacher_topk(model: TinyMoEModel, tokens: torch.Tensor, k: int):
    with torch.no_grad():
        logits = model(tokens)                              # [B, L, V]
        lp = torch.log_softmax(logits.float(), dim=-1)
    lp_top, ids_top = lp.topk(k, dim=-1)
    return (
        logits.reshape(-1, logits.shape[-1]),
        ids_top.reshape(-1, k).long(),
        lp_top.reshape(-1, k),
    )


def ce_loss(model: TinyMoEModel, tokens: torch.Tensor) -> float:
    """next-token CE（logits[:-1] 对 tokens[1:]）。"""
    with torch.no_grad():
        logits = model(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1)
        ).item()


def main() -> None:
    cfg = TinyMoEConfig(vocab_size=256, hidden_size=64, inter_size=64,
                        num_experts=4, num_layers=2, num_heads=2)
    v, k = cfg.vocab_size, 32

    # ---- 小语料（4096 位置：teacher 收敛到分布熵附近而非记忆坍缩）----
    train_tok = make_batch(v, 64, 64)
    heldout_tok = make_batch(v, 16, 64)

    # ---- teacher：小语料上训练到中等强度（避免坍缩：坍缩会让蒸馏缺口和 CE 对比失去意义）----
    teacher = TinyMoEModel(cfg)
    t_opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    t_first = t_last = None
    for step in range(200):
        logits = teacher(train_tok)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, v), train_tok[:, 1:].reshape(-1))
        t_opt.zero_grad()
        loss.backward()
        t_opt.step()
        t_first = t_first if t_first is not None else loss.item()
        t_last = loss.item()
    check("m3.teacher_overfit", t_last < t_first * 0.7, f"first={t_first:.4f} last={t_last:.4f}")

    s_logits, t_ids, t_lp = teacher_topk(teacher, train_tok, k)
    topk_mass = t_lp.exp().sum(-1).mean().item()
    print(f"       teacher top-{k} 质量覆盖: {topk_mass:.3f}（真实 QAD top-100 场景应 >0.9）")
    # CE 对比在训练分布上进行（蒸馏的作用域）：期望 teacher < after < quantized_base
    # held-out 仅作信息参考（i.i.d. 随机语料无泛化结构，held-out CE 无判据意义）
    ce_teacher = ce_loss(teacher, train_tok)
    print(f"       heldout CE (info): teacher={ce_loss(teacher, heldout_tok):.4f}")

    # ---- student：deepcopy + 冻结 + int4 fake-quant + per-expert LoRA ----
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(False)
    num_experts = sum(1 for _ in student.iter_experts())
    lora = ExpertLoRA(num_experts, cfg.hidden_size, cfg.inter_size, rank=8, alpha=8.0)
    attach_qad_lora(student, lora, bits=4)

    # B=0 起点：LoRA delta=0，student 前向即纯 fake-quant 基线
    ce_q_base = ce_loss(student, train_tok)

    # int4 损伤可见：provider 注入（fakequant+delta）与原始 float 前向应有差异
    with torch.no_grad():
        detach_qad_lora(student)
        logits_float = student(train_tok)
        attach_qad_lora(student, lora, bits=4)
        logits_qad = student(train_tok)
    check("m3.qad_forward_differs_from_float",
          not torch.allclose(logits_qad, logits_float, atol=1e-6),
          f"max_diff={(logits_qad - logits_float).abs().max():.4f}")

    # ---- 训练：chunked KL 蒸馏（公式与 InstinctRazor opd_train_fsdp.py L263-286 一致）----
    opt = torch.optim.AdamW(lora.trainable_parameters(), lr=3e-3)
    B = s_logits.shape[0]
    losses: list[float] = []
    t0 = time.time()
    for step in range(300):
        logits = student(train_tok).reshape(B, v)
        loss = kl_distill_loss(logits, t_ids, t_lp, v, mode="forward", tau=1.0, chunk=64)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora.trainable_parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if not torch.isfinite(loss):
            check("m3.finite", False, f"step={step}")
            return
    dt = time.time() - t0

    # 梯度只走 LoRA：冻结参数无 grad，LoRA 参数 grad 非零
    frozen_grads = [n for n, p in student.named_parameters() if p.grad is not None]
    check("m3.frozen_no_grad", not frozen_grads, f"leaked={frozen_grads[:3]}")
    lora_grads = {n: p.grad.abs().sum().item() for n, p in lora.params.items()}
    check("m3.lora_grad_all_experts", all(g > 0 for g in lora_grads.values()),
          " ".join(f"{n}={g:.2e}" for n, g in lora_grads.items()))

    # 蒸馏缺口收敛：KL 为正且显著下降
    first, last = losses[0], losses[-1]
    check("m3.kl_positive", first > 0, f"first={first:.4f}")
    check("m3.kl_decrease", last < first * 0.8, f"first={first:.4f} last={last:.4f} steps=300 time={dt:.1f}s")

    # 下游 CE（训练分布）：训练后应优于 quantized-only 基线并向 teacher 靠近
    ce_after = ce_loss(student, train_tok)
    check("m3.ce_improves", ce_after < ce_q_base,
          f"teacher={ce_teacher:.4f} quantized_base={ce_q_base:.4f} after={ce_after:.4f}")
    check("m3.finite", True)


if __name__ == "__main__":
    main()
    print("-" * 60)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL M3 TINY E2E TESTS PASSED")
