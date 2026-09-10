"""M4 中期诊断：训练分布上的 KL 对拍（区分"训练无效" vs "分布特化"）。

背景（2026-09-08 wikitext 探针结果）：QAD ckpt(1000/1500/2000) 在 wikitext 上
长窗持平、短窗略降 ~1%——但训练 KL loss 在降。本脚本在**训练语义 + 训练分布**
上直接对拍各臂 KL，判定训练是否真的降低了同分布 KL：
- 臂 1 baseline（B=0，= 无 QAD 的 fake-quant）
- 臂 2/3 ckpt 1000/2000
- 臂 4 skip-fq 校准（fake_quant 恒等 → 学生≈教师，KL 应≈0，验证管线）

语义 = 训练语义（无 attention W8、无激活 int8——与 ppl 探针的部署语义刻意区分）。
样本：全数据集均匀 stride（跨域代表性），shard 聚簇取数。

用法（容器内）：
    PYTHONPATH=/llm_infer_workspace/ylt/npuslim/src python \
        /llm_infer_workspace/ylt/npuslim/src/npuslim/qad/eval_qad_kl.py \
        --n-samples 24 --output /llm_infer_workspace/ylt/glm52-qad/eval_qad_kl.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/llm_infer_workspace/ylt/npuslim/src")
sys.path.insert(0, "/llm_infer_workspace/ylt/qwen3-qad/precompute")

from npuslim.qad.teacher_cache import TeacherCache            # noqa: E402
from npuslim.qad.kl_loss import kl_distill_loss               # noqa: E402
from npuslim.qad.ste_experts import wrap_all_experts          # noqa: E402
import npuslim.qad.ste_experts as _ste_exp                    # noqa: E402
from precompute_logits import encode_one                      # noqa: E402

MODEL_DIR = "/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507"
CACHE = Path("/llm_infer_workspace/ylt/qwen3-qad/precompute/logits_cache_topk100")
TRAIN = Path("/llm_infer_workspace/ylt/glm52-qad/qad_data/qad_train_v1.jsonl")
CORPUS = Path("/llm_infer_workspace/ylt/glm52-qad/glm52-qad-corpus/"
              "processed/corpus_v1_2026-08-27/clean")
VOCAB, VCHUNK = 151_936, 65536


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[
        "", "/llm_infer_workspace/ylt/glm52-qad/ckpt_m4/step_001000.pt",
        "/llm_infer_workspace/ylt/glm52-qad/ckpt_m4/step_002000.pt"])
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--min-enc-len", type=int, default=256)
    ap.add_argument("--micro-tokens", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--output", default="/llm_infer_workspace/ylt/glm52-qad/eval_qad_kl.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t0 = time.time()
    tc = TeacherCache(CACHE)
    entries = []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                entries.append((s["sample_key"], s["id"]))
    rng = random.Random(42)
    rng.shuffle(entries)
    # 均匀 stride 跨域取样 + shard 聚簇（fetch 局部性）
    stride = max(1, len(entries) // args.n_samples)
    pool = entries[::stride][:args.n_samples]
    order = {sid: (tc._by_id[sid][-1].shard, tc._by_id[sid][-1].row)
             for _, sid in pool if sid in tc._by_id}
    pool.sort(key=lambda e: order.get(e[1], ("", -1)))
    print(f"[kl] 样本 {len(pool)}/{len(entries)} stride={stride} "
          f"cache {time.time() - t0:.0f}s", flush=True)

    line_cache: dict[str, list[str]] = {}
    samples = []
    for key, sid in pool:
        fname, _, ln = key.rpartition(":")
        if fname not in line_cache:
            with open(CORPUS / fname, encoding="utf-8") as fh:
                line_cache[fname] = fh.readlines()
        row = json.loads(line_cache[fname][int(ln)].strip())
        enc = encode_one(tok, row, 8192)
        if not (enc is not None and not isinstance(enc, Exception) and len(enc) > 1):
            continue
        if len(enc) < args.min_enc_len:
            continue
        expected = np.asarray(enc[1:], dtype=np.int32)
        if expected.shape[0] < 20:
            continue
        fs = tc.fetch(sid, expected_ids=expected)
        L_enc = min(len(enc), args.micro_tokens)
        L_use = min(fs.topk_ids.shape[0], L_enc - 1)
        samples.append((np.asarray(enc, dtype=np.int64)[:L_enc],
                        fs.topk_ids[:L_use].copy(), fs.topk_lps[:L_use].copy()))
    print(f"[kl] 有效样本 {len(samples)}（取数 {time.time() - t0:.0f}s）", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    wrap_all_experts(model, rank=args.rank, vectorized=True,
                     dtype=torch.bfloat16, use_checkpoint=False, zero3_gather=False)
    print(f"[kl] 模型+STE 就绪 {time.time() - t0:.0f}s", flush=True)
    lora_params = {n: p for n, p in model.named_parameters() if ".lora.params." in n}

    def set_arm(ck: str):
        if not ck:
            with torch.no_grad():
                for n, p in lora_params.items():
                    if n.endswith(".Bgu") or n.endswith(".Bdn"):
                        p.zero_()
            return
        sd = torch.load(ck, map_location="cpu", weights_only=False)["sd"]
        with torch.no_grad():
            for k, v in sd.items():
                lora_params[k].data.copy_(v.to(lora_params[k].dtype))

    def run_kls() -> list[float]:
        vals = []
        with torch.no_grad():
            for inp_np, ids_np, lps_np in samples:
                inp = torch.from_numpy(inp_np).unsqueeze(0).to("npu")
                out = model(input_ids=inp, use_cache=False)
                lg = out.logits[0, :ids_np.shape[0]].float()
                ids_t = torch.from_numpy(ids_np).long().to(lg.device)
                lps_t = torch.from_numpy(lps_np).float().to(lg.device)
                loss = kl_distill_loss(lg, ids_t, lps_t, VOCAB, tau=1.0, chunk=VCHUNK)
                if torch.isfinite(loss):
                    vals.append(float(loss.item()))
        return vals

    results = []
    for ck in args.ckpts:
        tag = "baseline_lora0" if not ck else Path(ck).stem
        set_arm(ck)
        vals = run_kls()
        results.append({"arm": tag, "ckpt": ck, "n": len(vals),
                        "kl_mean": round(float(np.mean(vals)), 5),
                        "kl_median": round(float(np.median(vals)), 5),
                        "kl_values": [round(v, 5) for v in vals]})
        print(f"[kl] {tag}: mean={np.mean(vals):.5f} median={np.median(vals):.5f} "
              f"n={len(vals)} ({time.time() - t0:.0f}s)", flush=True)

    # 校准臂：fake_quant 恒等 → 学生≈教师（attention/激活均 bf16），KL 应≈0
    _ste_exp.fake_quant_weight = lambda w, bits=4, scale=None: w
    set_arm("")
    vals = run_kls()
    results.append({"arm": "skipfq_calibration", "n": len(vals),
                    "kl_mean": round(float(np.mean(vals)), 5),
                    "kl_median": round(float(np.median(vals)), 5)})
    print(f"[kl] skipfq_calibration: mean={np.mean(vals):.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  ensure_ascii=False, indent=2)
    print(f"[kl] 完成 -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
