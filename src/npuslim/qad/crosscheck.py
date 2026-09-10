"""M2 对拍集生成器 + fp64 参考实现（QAD Phase 1，2026-09-02）。

为 `kl_distill_loss` 的 NPU 端数值对拍制造标准输入：分布形态可控、
两端逐位一致（同一 npz）、带 fp64 dense 真值。分工：

- 阶段 generate（CPU，本文件）：36 用例（4 分布族 × B{1,8,64} × tau{0.5,1,2}）
  落盘 npz + fp64 loss/grad 真值
- 阶段 local（CPU，runner）：kl_distill_loss fp32 跑同输入 vs fp64 真值——
  把"实现 bug"与"设备差异"解耦
- 阶段 npu（NPU 恢复后，runner `--device npu`）：同 npz 对拍（判据 loss <1e-3、
  grad cos sim >0.999）+ chunk 显存标定

分布族设计（激活 kl_distill_loss 的四条数值路径，见调研文档 §6.2 M2）：
- teacher 自洽构造：top-K log-space 归一 + 显式 rest 质量，logprobs 是合法的
  log_softmax 截断值（与 845GB 缓存同性质）
- student：族幅值 randn + 样本级对齐混合（aligned 前半 / misaligned 后半，
  s_lp 分别 ~[-1,-5] 与 ~-12），覆盖 q_rest 与梯度的两种形态
- 边界注入：id=0/V-1（5%）与重复 id（5%）
- grad 真值仅存 B<=8（数值路径与 B 无关，B=64 用例为 NPU chunk 显存标定服务）

fp64 参考实现与 kl_loss.py 逐项同构，仅 dtype/分块不同——fp64 下两者数学
等价，故误差 ~1e-12；这是对分块实现最强的独立验证（无需 InstinctRazor GPU）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

K = 100           # teacher top-K

# 词表口径：GLM-5.2 缓存 154,880；Qwen3-30B（M4 首跑 teacher/学生）151,936。
# KL 实现 vocab 无关，两组用例共享同一数值路径，Qwen3 组为 NPU 端对拍/标定补齐。
GLM_VOCAB = 154_880
QWEN3_VOCAB = 151_936

# 分布族形状参数：(top-K log 权重衰减形式, rest 质量, student 幅值)
FAMILY_PARAMS = {
    #  sharp: p 从 ~1 衰到 1e-16，路径①主导，log_rest 靠 clamp 下界
    "sharp": dict(w_fn=lambda i: math.exp(-0.35 * i), rest=1e-6, amp=3.0),
    #  flat: 100 个等概率 token，路径②主导（rest 均摊项）
    "flat": dict(w_fn=None, rest=0.5, amp=1.0),   # w ~ lognormal(-4.6, 0.3)
    #  tail: 平缓衰减，①② 混合（真实 LLM 常见形态）
    "tail": dict(w_fn=lambda i: math.exp(-0.12 * i), rest=0.1, amp=2.0),
    #  large: 大幅值 student（amp 30 × tau 0.5 → z ±60），路径③ lse 极限压力
    "large": dict(w_fn=lambda i: math.exp(-0.35 * i), rest=1e-8, amp=30.0),
}
BATCHES = (1, 8, 64)
TAUS = (0.5, 1.0, 2.0)
GRAD_MAX_BATCH = 8   # grad 真值只存 B<=8（数值路径与 B 无关）


def _fam_seed(family: str) -> int:
    """族名 → 确定性 int（禁用 hash()：PYTHONHASHSEED 进程间不一致）。"""
    return int.from_bytes(family.encode(), "little")


@dataclass(frozen=True)
class CaseSpec:
    family: str
    batch: int
    tau: float
    vocab: int = GLM_VOCAB


def all_cases(vocab: int = GLM_VOCAB,
              batches: tuple[int, ...] = BATCHES,
              taus: tuple[float, ...] = TAUS) -> list[CaseSpec]:
    return [CaseSpec(f, b, t, vocab) for f in FAMILY_PARAMS for b in batches for t in taus]


def _teacher_topk(family: str, rng: np.random.Generator,
                  vocab: int = GLM_VOCAB) -> tuple[np.ndarray, np.ndarray]:
    """构造自洽的 teacher top-K：ids + log_softmax 截断值（[K]）。

    logprobs = log(w_i / Σw · (1−rest))，与 rest 质量严格自洽（Σp + rest = 1），
    与真实缓存的 log_softmax 截断值同性质。边界注入：10% 样本把 ids[0] 换成
    0 或 V−1。
    """
    par = FAMILY_PARAMS[family]
    ids = rng.choice(vocab, size=K, replace=False)
    if rng.random() < 0.1:                                   # 极值 id 注入
        ids[0] = 0 if rng.random() < 0.5 else vocab - 1
    if par["w_fn"] is None:                                  # flat: lognormal
        w = rng.lognormal(-4.6, 0.3, size=K)
    else:
        w = np.array([par["w_fn"](i) for i in range(K)])
    p = w / w.sum() * (1.0 - par["rest"])
    return ids, np.log(p)


def generate_case(spec: CaseSpec) -> dict[str, np.ndarray]:
    """生成单用例输入（不落盘）。样本级对齐混合：前半 aligned / 后半 misaligned。"""
    par = FAMILY_PARAMS[spec.family]
    rng = np.random.default_rng(
        _fam_seed(spec.family) * 1_000_003 + spec.batch + spec.vocab)

    ids = np.empty((spec.batch, K), dtype=np.int64)
    lps = np.empty((spec.batch, K), dtype=np.float32)
    for i in range(spec.batch):
        ids[i], lps[i] = _teacher_topk(spec.family, rng, spec.vocab)
    # 重复 id 注入（gather 重复位置的合法路径；参考实现同公式自动一致）
    dup_rows = rng.random(spec.batch) < 0.05
    for i in np.where(dup_rows)[0]:
        ids[i, 1] = ids[i, 0]

    g = torch.Generator().manual_seed(
        _fam_seed(spec.family) * 1_000_003 + spec.batch * 1000
        + round(spec.tau * 10) + spec.vocab)
    logits = torch.randn(spec.batch, spec.vocab, generator=g) * par["amp"]
    aligned = spec.batch // 2
    for i in range(aligned):                                 # aligned: ids 位置递减偏置
        boost = 4.0 * torch.exp(-0.05 * torch.arange(K, dtype=torch.float32))
        logits[i, torch.from_numpy(ids[i].astype(np.int64))] += boost
    return {
        "student_logits": logits.numpy().astype(np.float32),
        "teacher_topk_ids": ids,
        "teacher_topk_logprobs": lps,
    }


def fp64_reference(case: dict[str, np.ndarray], vocab_size: int, tau: float,
                   compute_grad: bool = True) -> tuple[float, np.ndarray | None]:
    """fp64 dense 参考（与 kl_loss.py 逐项同构，仅 dtype/分块不同）。

    返回 (loss, grad)；grad 为对 student_logits 的 fp32 梯度（1/tau chain rule
    由 autograd 自动处理），compute_grad=False 时返回 (loss, None)。
    """
    logits = torch.from_numpy(case["student_logits"]).double()
    if compute_grad:
        logits.requires_grad_(True)
    ids = torch.from_numpy(case["teacher_topk_ids"])
    log_p = torch.from_numpy(case["teacher_topk_logprobs"]).double()

    z = logits / tau
    lse = torch.logsumexp(z, dim=-1)                         # fp64 dense，无分块
    s_lp = z.gather(-1, ids) - lse.unsqueeze(-1)
    p = log_p.exp()
    rest = (1.0 - p.sum(dim=-1)).clamp_min(0.0)
    log_rest = torch.log(rest.clamp_min(1e-12)) - math.log(vocab_size - K)
    q = s_lp.exp()
    q_sum = q.sum(dim=-1).clamp_max(1.0 - 1e-7)
    q_rest = 1.0 - q_sum
    log_q_rest = torch.log(q_rest.clamp_min(1e-12))
    loss = ((p * (log_p - s_lp)).sum(dim=-1) + rest * (log_rest - log_q_rest)).mean()

    grad = None
    if compute_grad:
        loss.backward()
        grad = logits.grad.to(torch.float32).numpy()
    return float(loss.item()), grad


def generate_all(out_dir: Path, vocab: int = GLM_VOCAB,
                 batches: tuple[int, ...] = BATCHES,
                 taus: tuple[float, ...] = TAUS,
                 extra_specs: tuple[CaseSpec, ...] = ()) -> Path:
    """生成用例网格 + fp64 真值，落盘 npz + index.json，返回 index 路径。

    extra_specs 允许追加网格外的单个用例（如 Qwen3 组的 B=64 显存标定用例）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in list(all_cases(vocab, batches, taus)) + list(extra_specs):
        case = generate_case(spec)
        loss64, grad = fp64_reference(
            case, spec.vocab, spec.tau, compute_grad=spec.batch <= GRAD_MAX_BATCH)
        name = f"m2_{spec.family}_B{spec.batch}_tau{spec.tau:g}.npz"
        payload = {k: v for k, v in case.items()}
        payload["ref_loss"] = np.float64(loss64)
        if grad is not None:
            payload["ref_grad"] = grad
        np.savez_compressed(out_dir / name, **payload)
        index.append({"file": name, **asdict(spec), "ref_loss": loss64,
                      "grad_stored": grad is not None})
        print(f"       {name}  ref_loss={loss64:+.6e}")
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(
        {"vocab": vocab, "k": K, "cases": index}, indent=1))
    return index_path
