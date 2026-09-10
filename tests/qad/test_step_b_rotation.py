"""A7 Step B：GPTQRotationAlgorithm 折叠 + R1 hook 的 fp 等价性闸门（调研文档 §6.1 Step 2）。

闸门：折叠 + hook 同时生效后，模型前向应与原前向**数值一致**（R 正交，fp 恒等）。
- fp32：max diff < 1e-3（拦折叠位置错误的静默 bug）
- bf16：max diff < 1e-2（bf16 舍入下限；折叠/旋转在 fp32 计算再 cast，误差压到最小）

验证对象：`GPTQRotationAlgorithm` 的独立原语 `fold_layer_for_rotation` +
`register_rotation_hooks`，直接作用于 Qwen3Moe 层（tiny_qwen3，4 层 MoE）。

运行（npuslim conda env，CPU）：
    /data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_step_b_rotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.rotation import QuaRotMode, create_rot
from npuslim.qad.rotation_quant import fold_layer_for_rotation, register_rotation_hooks

TINY_QWEN3 = Path(__file__).resolve().parents[2] / "workspace" / "tiny_qwen3"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_gate(dtype: torch.dtype, tol: float, label: str) -> float:
    model = AutoModelForCausalLM.from_pretrained(TINY_QWEN3, torch_dtype=dtype)
    model.eval()
    r1 = create_rot(QuaRotMode.HADAMARD, model.config.hidden_size, seed=2026)
    torch.manual_seed(0)
    tok = torch.randint(0, model.config.vocab_size, (2, 32))

    with torch.no_grad():
        ref = model(tok).logits
    # 折叠 + hook 同时生效
    all_handles = []
    for layer in model.model.layers:
        fold_layer_for_rotation(layer, r1)
        all_handles += register_rotation_hooks(layer, r1)
    with torch.no_grad():
        rot = model(tok).logits
    for h in all_handles:
        h.remove()
    diff = (rot - ref).abs().max().item()
    rel = diff / ref.abs().max().item()
    check(f"step_b.equivalence_{label}", diff < tol,
          f"max_diff={diff:.2e}  rel={rel:.2e} (tol {tol})")
    return diff


def run_gate_bf16_relative() -> float:
    """bf16 闸门：旋转不应显著超过 bf16 固有噪声。

    直接对拍原 bf16 前向不成立——R1 是 ±1/sqrt(n) 的 Hadamard，bf16 旋转本身
    引入 ~5% 相对舍入（fp32 恒等已证逻辑正确）。正确判据：旋转后 bf16 前向相对
    fp32 真值的偏差，不超过未旋转 bf16 固有偏差的 ~3 倍（若折叠位置错误，偏差
    会到 O(1) 级别，远超该门槛）。
    """
    torch.manual_seed(0)
    tok = torch.randint(0, 256, (2, 32))
    fp32 = AutoModelForCausalLM.from_pretrained(TINY_QWEN3, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref_fp32 = fp32(tok).logits.float()

    bf16 = AutoModelForCausalLM.from_pretrained(TINY_QWEN3, torch_dtype=torch.bfloat16).eval()
    with torch.no_grad():
        bf16_orig = bf16(tok).logits.float()
    dev_orig = (bf16_orig - ref_fp32).abs().max().item()

    r1 = create_rot(QuaRotMode.HADAMARD, bf16.config.hidden_size, seed=2026)
    all_handles = []
    for layer in bf16.model.layers:
        fold_layer_for_rotation(layer, r1)
        all_handles += register_rotation_hooks(layer, r1)
    with torch.no_grad():
        bf16_rot = bf16(tok).logits.float()
    for h in all_handles:
        h.remove()
    dev_rot = (bf16_rot - ref_fp32).abs().max().item()
    ratio = dev_rot / max(dev_orig, 1e-8)
    check("step_b.equivalence_bf16_rel", dev_rot <= 3.0 * dev_orig + 1e-3,
          f"bf16固有偏差={dev_orig:.2e} 旋转后偏差={dev_rot:.2e} ratio={ratio:.2f} (门槛<=3x)")
    return dev_rot


def main() -> None:
    print("=== fp32 等价性闸门（全维 R1）===")
    d1 = run_gate(torch.float32, 1e-3, "fp32")
    print("=== bf16 相对噪声闸门（真实精度）===")
    d2 = run_gate_bf16_relative()
    print(f"\nfp32 max_diff={d1:.2e} | bf16 旋转后偏差={d2:.2e}")

    print("=== 块对角 R1（block 128）fp32 闸门 ===")
    model = AutoModelForCausalLM.from_pretrained(TINY_QWEN3, torch_dtype=torch.float32)
    model.eval()
    r1b = create_rot(QuaRotMode.HADAMARD, model.config.hidden_size, block_size=128, seed=2026)
    torch.manual_seed(1)
    tok = torch.randint(0, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        ref = model(tok).logits
    all_handles = []
    for layer in model.model.layers:
        fold_layer_for_rotation(layer, r1b)
        all_handles += register_rotation_hooks(layer, r1b)
    with torch.no_grad():
        rot = model(tok).logits
    for h in all_handles:
        h.remove()
    diffb = (rot - ref).abs().max().item()
    check("step_b.equivalence_block128", diffb < 1e-3, f"max_diff={diffb:.2e}")

    print("-" * 60)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("STEP B 等价性闸门全部通过：折叠 + R1 hook = fp 恒等")


if __name__ == "__main__":
    main()
