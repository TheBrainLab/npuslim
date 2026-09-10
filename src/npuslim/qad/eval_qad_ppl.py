"""M4 中期探针：QAD LoRA ckpt 的 wikitext-2 ppl（部署语义，多臂单次加载）。

与 §9.3 基线（GPTQ W4A8 NOROT 11.9525，32 窗口 × 256 seq）语义对齐：
- 专家：STE fake-quant（训练同款 fake_quant_weight bits=4，bf16 网格）+ QAD LoRA delta
  （output-space，delta 加在 fake-quant 之后——与 ste_experts._effective_all 一致）
- 稠密 q/k/v/o：per-channel 对称 int8 fake-quant（W8A8 部署语义，直接改写冻结权重）
- 激活：input/post layernorm 输出 per-token int8 动态 fake-quant
  （复用 eval_rotated_ppl.register_eval_hooks，无旋转）
- norm/embed/router/lm_head：bf16 原样

多臂设计：模型只加载/包装一次，按序评测各臂（省 61GB 模型的重复加载）：
  臂 1 = LoRA=0 基线（B 恒 0 → delta 恒 0 = "无 QAD 的 fake-quant PTQ"）
  臂 2+ = --ckpts 各断点（sd 覆盖全部 192 张量，臂间无泄漏）
每臂跑两种窗口：32×256（与基线可比）+ 16×1024（更高信噪比看趋势）。

rank 自动从首个 ckpt 的 args 读取（训练 --rank 16 → scaling=64/16=4）。

用法（容器内单进程；device_map 分摊多 NPU，单卡 bf16 61GB+LoRA 太紧）：
    python -m npuslim.qad.eval_qad_ppl \
        --model-path /home/yult/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507 \
        --ckpts "" /home/yult/llm_infer_workspace/ylt/glm52-qad/ckpt_m4/step_001000.pt \
                  /home/yult/llm_infer_workspace/ylt/glm52-qad/ckpt_m4/step_002000.pt \
        --wikitext-dir /home/yult/llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1 \
        --device-map auto --output /home/yult/llm_infer_workspace/ylt/glm52-qad/eval_qad_mid.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from npuslim.qad.ste_experts import wrap_all_experts                      # noqa: E402
from npuslim.qad.eval_rotated_ppl import register_eval_hooks, eval_ppl    # noqa: E402


def fake_quant_w8_perchannel(w: torch.Tensor) -> torch.Tensor:
    """稠密层 W8A8 权重模拟：per-output-channel 对称 int8（rel_err ~1.1% 同量级）。"""
    s = w.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return (w.float() / s).round().clamp(-127, 127).mul(s).to(w.dtype)


def apply_dense_w8(model) -> int:
    """q/k/v/o 冻结权重原地 int8 fake-quant，返回改写模块数。"""
    n = 0
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                ref = getattr(layer.self_attn, proj).weight
                ref.data = fake_quant_w8_perchannel(ref.data)
                n += 1
    return n


def set_lora_state(model, sd: dict | None) -> tuple[int, int]:
    """装载 LoRA sd（None = 基线臂：B 全零 → delta 恒 0）。返回 (装载, 缺失) 数。"""
    lora_params = {n: p for n, p in model.named_parameters() if ".lora.params." in n}
    if sd is None:
        with torch.no_grad():
            for n, p in lora_params.items():
                if n.endswith(".Bgu") or n.endswith(".Bdn"):
                    p.zero_()
        return 0, 0
    missing = [k for k in sd if k not in lora_params]
    got = 0
    with torch.no_grad():
        for k, v in sd.items():
            if k in lora_params:
                lora_params[k].data.copy_(v.to(lora_params[k].dtype))
                got += 1
    return got, len(missing)


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 中期探针：QAD ckpt wikitext ppl（部署语义）")
    ap.add_argument("--model-path",
                    default="/home/yult/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--ckpts", nargs="*", default=[""],
                    help='断点列表；"" 表示 LoRA=0 基线臂，其余为 ckpt_*.pt 路径')
    ap.add_argument("--wikitext-dir",
                    default="/home/yult/llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1")
    ap.add_argument("--device-map", default="auto",
                    help='auto=多 NPU 分摊；单卡装不下 61GB bf16 + LoRA')
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=32)   # 与 §9.3 基线一致
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--big-windows", type=int, default=16)
    ap.add_argument("--big-seq", type=int, default=1024)
    ap.add_argument("--output", default="eval_qad_mid.json")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[probe] 加载 BF16 模型 (device_map={args.device_map}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=args.device_map, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[probe] 模型加载完成 {time.time() - t0:.0f}s", flush=True)

    # rank 从首个真实 ckpt 读取（训练 --rank 决定 scaling=64/rank）
    rank = 16
    for c in args.ckpts:
        if c:
            rank = int(torch.load(c, map_location="cpu", weights_only=False)
                       ["args"].get("rank", 16))
            break
    print(f"[probe] STE 包装（rank={rank}, scaling={64.0 / rank}）...", flush=True)
    wrapped = wrap_all_experts(model, rank=rank, vectorized=True,
                               use_checkpoint=False, zero3_gather=False)
    n_w8 = apply_dense_w8(model)
    handles = register_eval_hooks(model, r1=None, act_bits=args.act_bits)
    print(f"[probe] 稠密 W8 改写 {n_w8} 模块 + 激活 int{args.act_bits} hook 就绪 "
          f"{time.time() - t0:.0f}s", flush=True)

    device = model.model.embed_tokens.weight.device
    results = []
    for ck in args.ckpts:
        tag = "baseline_lora0" if not ck else Path(ck).stem
        sd = None if not ck else torch.load(ck, map_location="cpu",
                                            weights_only=False)["sd"]
        got, n_miss = set_lora_state(model, sd)
        if ck and (got != len(sd) or n_miss):
            print(f"[probe][{tag}] WARN 装载 {got}/{len(sd)} 缺失 {n_miss}", flush=True)
        p1 = eval_ppl(model, tok, n_windows=args.n_windows, seq=args.seq,
                      device=device, wikitext_dir=args.wikitext_dir)
        p2 = eval_ppl(model, tok, n_windows=args.big_windows, seq=args.big_seq,
                      device=device, wikitext_dir=args.wikitext_dir)
        results.append({"arm": tag, "ckpt": ck,
                        "ppl_32x256": round(p1, 4), "ppl_16x1024": round(p2, 4)})
        print(f"[probe] {tag}: ppl(32x256)={p1:.4f}  ppl(16x1024)={p2:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    for h in handles:
        h.remove()
    out = {"model": args.model_path, "rank": rank, "act_bits": args.act_bits,
           "results": results, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[probe] 完成，结果写入 {args.output}")


if __name__ == "__main__":
    main()
