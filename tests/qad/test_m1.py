"""M1 单元测试：STE fake-quant / chunked KL loss / per-expert LoRA / mini 端到端。

运行（npuslim conda env，CPU 即可）：
    /data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_m1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.kl_loss import _chunked_logsumexp, kl_distill_loss
from npuslim.qad.lora import ExpertLoRA
from npuslim.qad.ste import compute_weight_scale, fake_quant_weight

torch.manual_seed(0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------- M1.1 STE ----------
def test_ste() -> None:
    w = torch.randn(64, 128) * 0.05
    for bits, (qmin, qmax) in ((4, (-8, 7)), (8, (-128, 127))):
        s = compute_weight_scale(w, bits)
        check(f"ste.scale_shape[{bits}]", s.shape == (64, 1) and bool((s > 0).all()))
        wq = fake_quant_weight(w, bits)
        q = torch.round(w / s)
        check(f"ste.grid[{bits}]", bool((q >= qmin).all() and (q <= qmax).all())
              and bool(torch.allclose(wq, q.clamp(qmin, qmax) * s)))
        # 梯度直通：d(out)/d(w) 恒等
        w_leaf = w.clone().requires_grad_(True)
        fake_quant_weight(w_leaf, bits).sum().backward()
        check(f"ste.grad_identity[{bits}]", bool(torch.equal(w_leaf.grad, torch.ones_like(w))))
    # int4 网格与部署补码解码一致：q 与 (nibble 补码) 相同
    w = torch.randn(16, 32) * 0.03
    s = compute_weight_scale(w, 4)
    q = torch.round(w / s).clamp(-8, 7)
    nib = (q.to(torch.int8) & 0x0F).to(torch.int8)
    nib_tc = torch.where(nib >= 8, nib - 16, nib).float()
    check("ste.twos_complement", bool(torch.allclose(q, nib_tc)))


# ---------- M1.2 chunked KL ----------
def _ref_forward_kl(student: torch.Tensor, t_ids: torch.Tensor, t_lp: torch.Tensor,
                    v: int, k: int, tau: float) -> torch.Tensor:
    """fp64 dense 参考：与 kl_distill_loss 同一定义（InstinctRazor 公式），
    逐样本全量算术展开，验证 chunked 实现的数值正确性（而非逼近差）。"""
    z = student.double() / tau
    log_q = torch.log_softmax(z, dim=-1)                     # [B, V]
    s_lp = log_q.gather(1, t_ids.long())                     # [B, K]
    log_p = t_lp.double()
    p = torch.exp(log_p)
    rest = 1.0 - p.sum(dim=-1)
    log_rest = torch.log(rest.clamp_min(1e-300)) - torch.log(torch.tensor(float(v - k)))
    q = torch.exp(s_lp)
    log_q_rest = torch.log((1.0 - q.sum(dim=-1)).clamp_min(1e-300))
    kl = (p * (log_p - s_lp)).sum(dim=-1) + rest * (log_rest - log_q_rest)
    return kl.mean()


def test_kl() -> None:
    b, v, k = 7, 4096, 100
    student = torch.randn(b, v) * 3.0
    t_ids = torch.topk(student + torch.randn(b, v) * 0.5, k, dim=-1).indices
    z_t = student.double() + torch.randn(b, v).double() * 0.2
    logp_t = torch.log_softmax(z_t, dim=-1)
    t_lp = logp_t.gather(-1, t_ids)

    # chunked logsumexp 对拍 torch.logsumexp
    lse_c = _chunked_logsumexp(student, chunk=512)
    lse_r = torch.logsumexp(student.double(), dim=-1).float()
    check("kl.chunked_lse", bool((lse_c - lse_r).abs().max() < 1e-4),
          f"max_err={(lse_c - lse_r).abs().max():.2e}")

    for tau in (1.0, 0.5):
        loss = kl_distill_loss(student, t_ids, t_lp, v, mode="forward", tau=tau, chunk=512)
        ref = _ref_forward_kl(student, t_ids, t_lp, v, k, tau)
        check(f"kl.forward_vs_dense[tau={tau}]", bool((loss - ref).abs() < 1e-3),
              f"loss={loss:.6f} ref={ref:.6f} err={abs(loss - ref):.2e}")

    rev = kl_distill_loss(student, t_ids, t_lp, v, mode="reverse", chunk=512)
    check("kl.reverse_finite", bool(torch.isfinite(rev)), f"rev={rev:.6f}")

    # 梯度流
    s_leaf = student.clone().requires_grad_(True)
    kl_distill_loss(s_leaf, t_ids, t_lp, v, chunk=512).backward()
    check("kl.grad_finite", bool(torch.isfinite(s_leaf.grad).all())
          and bool(s_leaf.grad.abs().sum() > 0))


# ---------- M1.3 LoRA ----------
def test_lora() -> None:
    e, h, i, r = 4, 32, 16, 8
    lora = ExpertLoRA(e, h, i, rank=r, alpha=r)
    d = lora.delta("gate_up", 0)
    check("lora.delta_zero_init", d.shape == (2 * i, h) and bool((d == 0).all()))
    check("lora.delta_down_shape", lora.delta("down", 0).shape == (h, i))
    lora.params["Bgu"].data[0].normal_(0, 0.01)
    check("lora.delta_after_update", bool((lora.delta("gate_up", 0) != 0).any()))
    check("lora.trainable_params", len(lora.trainable_parameters()) == 4)


# ---------- M1.5 mini 端到端 ----------
def test_e2e() -> None:
    """单专家训练冒烟：output-space LoRA（delta 加在 fakequant(W) 之后）
    对齐一个理想专家权重的 teacher 分布。验收：loss 下降、梯度流经 LoRA、无 NaN。"""
    torch.manual_seed(7)
    out_f, in_f, v, k, n_tok = 64, 48, 512, 50, 256
    w = torch.randn(out_f, in_f) * 0.02          # 待蒸馏的 fake 专家权重
    w_ideal = torch.randn(out_f, in_f) * 0.02    # teacher 理想专家
    x = torch.randn(n_tok, in_f)

    # teacher top-K（K+1 格式：列 0 = 真实 token，此处直接给 top-K 集合与 logprob）
    logits_ideal = x @ w_ideal.T
    lp_full = torch.log_softmax(logits_ideal.double() * 2.0, dim=-1)
    t_lp_full, t_ids_full = lp_full.topk(k, dim=-1)
    t_ids_full = t_ids_full.long()      # gather 需要 int64
    t_lp_full = t_lp_full.float()

    # LoRA：down 对的 delta 形状 [hidden=out_f, inter=in_f] -> 用作 [out_f, in_f] 增量
    lora = ExpertLoRA(1, hidden_size=out_f, moe_intermediate_size=in_f, rank=8, alpha=8.0)
    opt = torch.optim.AdamW(lora.trainable_parameters(), lr=1e-2)

    first = last = None
    for step in range(60):
        w_eff = fake_quant_weight(w, bits=4) + lora.delta("down", 0)
        logits = x @ w_eff.T
        loss = kl_distill_loss(logits, t_ids_full, t_lp_full, v, chunk=256)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora.trainable_parameters(), 1.0)
        opt.step()
        if not torch.isfinite(loss):
            check("e2e.finite", False, f"step={step} loss={loss}")
            return
        first = first if first is not None else loss.item()
        last = loss.item()
    check("e2e.loss_decrease", last < first * 0.9, f"first={first:.4f} last={last:.4f}")
    check("e2e.grad_flow", bool(lora.params["Bdn"].grad is not None
                                and lora.params["Bdn"].grad.abs().sum() > 0))
    check("e2e.finite", True)


if __name__ == "__main__":
    test_ste()
    test_kl()
    test_lora()
    test_e2e()
    print("-" * 60)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL M1 TESTS PASSED")
