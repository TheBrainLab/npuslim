#!/usr/bin/env python3
"""M5-3 四臂对拍（同口径）：RTN vs GPTQ 误差画像定案。

统一语义 = eval_qad_ppl 的部署模拟：稠密 W8 改写 + norm 输出 per-token int8 hook，
32×256 + 16×1024 双窗 wikitext。变量仅专家权重来源：
  arm bf16_experts : 原始 BF16 专家权重（无专家量化参照；稠密/激活模拟照常）
  arm gptq_awq     : 原部署产物（Hessian 补偿 + AWQ）反量化权重
  arm gptq_noawq   : D-2×2 noAWQ 产物反量化权重
RTN 臂（②）= eval_qad_ppl 的 baseline_lora0（同一探针核心，09-09 终期探针实测
  11.6250 / 8.3452，两次逐位复现），本脚本不重复跑。

用法（glm5_qad_env，4 卡）：
  python3 m53_fourarm.py \
    --base /home/yult/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507 \
    --product-awq /home/yult/llm_infer_workspace/ylt/models/qwen3_30b_a3b_2507-w4a8_npu \
    --product-noawq /home/yult/llm_infer_workspace/ylt/npuslim/outputs_npu/stepB/norot_noawq \
    --output /home/yult/llm_infer_workspace/ylt/glm52-qad/eval_m53_fourarm.json
"""
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
NPUSLIM_SRC = _pd_os.environ.get("NPUSLIM_SRC", YLT + "/npuslim/src")
QAD_ROOT = _pd_os.environ.get("QAD_ROOT", YLT + "/glm52-qad")
# =========================================================

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from npuslim.qad.eval_qad_ppl import apply_dense_w8
from npuslim.qad.eval_rotated_ppl import (
    dequant_checkpoint,
    eval_ppl,
    rebuild_model_weights,
    register_eval_hooks,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--product-awq", required=True)
    ap.add_argument("--product-noawq", required=True)
    ap.add_argument("--wikitext-dir",
                    default=f"{SHARED_HOST}/ylt/npuslim/workspace/wikitext-2-raw-v1")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--big-windows", type=int, default=16)
    ap.add_argument("--big-seq", type=int, default=1024)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[m53] 加载 BF16 模型 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map=args.device_map, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print(f"[m53] 模型加载完成 {time.time()-t0:.0f}s", flush=True)

    n_w8 = apply_dense_w8(model)
    handles = register_eval_hooks(model, r1=None, act_bits=args.act_bits)
    print(f"[m53] 稠密 W8 改写 {n_w8} 模块 + 激活 int{args.act_bits} hook 就绪", flush=True)

    device = model.model.embed_tokens.weight.device
    results = []

    def run_arm(tag: str) -> None:
        t = time.time()
        p1 = eval_ppl(model, tok, n_windows=args.n_windows, seq=args.seq,
                      device=device, wikitext_dir=args.wikitext_dir)
        p2 = eval_ppl(model, tok, n_windows=args.big_windows, seq=args.big_seq,
                      device=device, wikitext_dir=args.wikitext_dir)
        results.append({"arm": tag, "ppl_32x256": round(p1, 4),
                        "ppl_16x1024": round(p2, 4)})
        print(f"[m53] {tag}: ppl(32x256)={p1:.4f}  ppl(16x1024)={p2:.4f}  "
              f"({time.time()-t:.0f}s)", flush=True)

    # 臂①：BF16 专家权重（稠密 W8 + 激活模拟照常）
    run_arm("bf16_experts")

    # 臂③/④：产物反量化 → 重建 → 评测（稠密误差来自产物本身，不再二次模拟）
    for tag, pdir in (("gptq_awq", args.product_awq),
                      ("gptq_noawq", args.product_noawq)):
        print(f"[m53] dequant + rebuild: {pdir} ...", flush=True)
        deq = dequant_checkpoint(pdir)
        rebuild_model_weights(model, deq)
        deq.clear()
        gc.collect()
        run_arm(tag)

    for h in handles:
        h.remove()

    out = {"meta": {"base": args.base,
                    "semantics": "dense W8 rewrite + per-token act int8 hooks (统一部署模拟)",
                    "rtn_arm_source": "eval_qad_ppl baseline_lora0 @ 09-09 终期探针"},
           "arms": results}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[m53] DONE -> {args.output}", flush=True)
    for r in results:
        print(f"  {r['arm']:<14} {r['ppl_32x256']:.4f} / {r['ppl_16x1024']:.4f}")


if __name__ == "__main__":
    main()
