"""M2 对拍 runner：generate / local / npu 三阶段（§6.2 M2 对拍步骤 1-5）。

用法（npuslim conda env）：
    python tests/qad/test_m2_crosscheck.py generate glm     # GLM vocab=154,880 全网格（36 用例）
    python tests/qad/test_m2_crosscheck.py generate qwen3   # Qwen3 vocab=151,936（24+1 用例）
    python tests/qad/test_m2_crosscheck.py local [glm|qwen3]   # CPU 对拍 vs fp64 真值
    python tests/qad/test_m2_crosscheck.py npu [glm|qwen3]     # NPU 对拍 + chunk 显存标定
                                                               #（步骤 3-5，NPU 恢复后执行）
判据（§6.2 M2 表）：
- local：loss |Δ| ~1e-5、grad cos sim >0.9999（纯精度差，无设备差）
- npu：  loss |Δ| <1e-3、grad cos sim >0.999（计划标准，容忍 bf16/fp32 舍入）
数据目录：workspace/glm52-qad/m2_crosscheck{,_qwen3}/（fp64 真值在 npz 内）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.crosscheck import CaseSpec, QWEN3_VOCAB, generate_all
from npuslim.qad.kl_loss import kl_distill_loss

_BASE = Path(__file__).resolve().parents[2] / "workspace" / "glm52-qad"
DATASETS = {                      # 数据集名 → 目录 + 生成参数（词表口径见 crosscheck.py）
    "glm":   (_BASE / "m2_crosscheck",          dict()),
    "qwen3": (_BASE / "m2_crosscheck_qwen3",    dict(
        vocab=QWEN3_VOCAB,
        batches=(1, 8),                         # 数值对拍网格
        taus=(0.5, 1.0, 2.0),
        extra_specs=(CaseSpec("large", 64, 0.5, QWEN3_VOCAB),))),  # B=64 仅标定用例
}
DATA_DIR = DATASETS["glm"][0]     # 当前数据集目录（main 里按 argv[2] 切换）

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def cmd_generate(name: str) -> None:
    out_dir, kwargs = DATASETS[name]
    t0 = time.time()
    idx = generate_all(out_dir, **kwargs)
    cases = json.loads(idx.read_text())["cases"]
    total = sum((out_dir / c["file"]).stat().st_size for c in cases)
    print(f"       {len(cases)} 用例 / {total / 1e6:.0f} MB / {time.time() - t0:.0f}s → {out_dir}")


def _load_cases() -> tuple[int, list[dict]]:
    idx = json.loads((DATA_DIR / "index.json").read_text())
    return int(idx["vocab"]), idx["cases"]


def cmd_local() -> None:
    """CPU fp32 chunked 实现 vs fp64 dense 真值——把实现 bug 与设备差异解耦。"""
    vocab, cases = _load_cases()
    for meta in cases:
        d = np.load(DATA_DIR / meta["file"])
        loss = kl_distill_loss(
            torch.from_numpy(d["student_logits"]),
            torch.from_numpy(d["teacher_topk_ids"]).long(),
            torch.from_numpy(d["teacher_topk_logprobs"]),
            vocab, tau=meta["tau"], chunk=32768).item()
        err = abs(loss - float(d["ref_loss"]))
        line = f"  [{meta['file']:<28}] loss={loss:+.6e}  |Δ|={err:.2e}"
        ok = err < 1e-4
        # 梯度对拍（B<=8 用例）：backward 后与 fp64 真值算 cos sim
        if "ref_grad" in d:
            lg = torch.from_numpy(d["student_logits"]).clone().requires_grad_(True)
            kl_distill_loss(lg, torch.from_numpy(d["teacher_topk_ids"]).long(),
                            torch.from_numpy(d["teacher_topk_logprobs"]),
                            vocab, tau=meta["tau"]).backward()
            ref = torch.from_numpy(d["ref_grad"])
            cos = torch.nn.functional.cosine_similarity(
                lg.grad.flatten().double(), ref.flatten().double(), dim=0).item()
            ok = ok and cos > 0.9999
            line += f"  cos={cos:.6f}"
        print(line + ("" if ok else "  ← FAIL"))
        check(f"local.{meta['file']}", ok)
        del d


def cmd_npu() -> None:
    """NPU 端对拍 + chunk 显存标定（步骤 3-5，NPU 恢复后执行）。"""
    try:
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    except ImportError:
        print("torch_npu 不可用——本阶段需 NPU 环境")
        sys.exit(2)
    vocab, cases = _load_cases()
    for meta in cases:
        d = np.load(DATA_DIR / meta["file"])
        logits = torch.from_numpy(d["student_logits"]).to("npu")
        ids = torch.from_numpy(d["teacher_topk_ids"]).long().to("npu")
        lps = torch.from_numpy(d["teacher_topk_logprobs"]).to("npu")

        lg = logits.clone().requires_grad_(True)
        kl_distill_loss(lg, ids, lps, vocab, tau=meta["tau"]).backward()
        torch.npu.synchronize()
        err = abs(kl_distill_loss(logits, ids, lps, vocab, tau=meta["tau"]).item()
                  - float(d["ref_loss"]))
        ok = err < 1e-3
        line = f"  [{meta['file']:<28}] |Δ|={err:.2e}"
        if "ref_grad" in d:
            cos = torch.nn.functional.cosine_similarity(
                lg.grad.flatten().float().cpu(),
                torch.from_numpy(d["ref_grad"]).flatten(), dim=0).item()
            ok = ok and cos > 0.999
            line += f"  cos={cos:.6f}"
        print(line + ("" if ok else "  ← FAIL"))
        check(f"npu.{meta['file']}", ok)

    # 步骤 5：chunk 显存标定（最大用例 B=64）
    calib = next(m for m in cases if m["batch"] == 64)
    d = np.load(DATA_DIR / calib["file"])
    logits = torch.from_numpy(d["student_logits"]).to("npu")
    print(f"       chunk 显存标定（{calib['file']}，含 autograd 保存量）:")
    for chunk in (8192, 16384, 32768, 65536):
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        lg = logits.clone().requires_grad_(True)
        loss = kl_distill_loss(lg, torch.from_numpy(d["teacher_topk_ids"]).long().to("npu"),
                               torch.from_numpy(d["teacher_topk_logprobs"]).to("npu"),
                               vocab, tau=calib["tau"], chunk=chunk)
        loss.backward()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() / 2**30
        print(f"       chunk={chunk:>6}: peak {peak:.2f} GiB")
    print("       → VCHUNK 取满足 64GB 预算（留 lm_head 梯度余量）的最大稳定值")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "local"
    if stage == "generate":
        cmd_generate(sys.argv[2] if len(sys.argv) > 2 else "glm")
    else:
        if len(sys.argv) > 2:
            DATA_DIR = DATASETS[sys.argv[2]][0]  # type: ignore[misc]
        {"local": cmd_local, "npu": cmd_npu}[stage]()
        print("-" * 76)
        if FAILURES:
            print(f"FAILED ({len(FAILURES)}): {FAILURES[:5]}")
            sys.exit(1)
        print(f"{stage.upper()} CROSSCHECK PASSED")
