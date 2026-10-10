"""单卡对照探针（无任何并行框架）：loss 异常根因判定。

缓存行语义：teacher 行 i = P(enc[i+1] | enc[0..i])，col0 = enc[i+1]，
actual_ids = enc[1:] 只是"对齐自校验流"，不能直接当学生输入——否则学生
logits 行 j 预测 enc[j+2] 却与 teacher 行 j（预测 enc[j+1]）比较 → off-by-one。

四基线一次加载测完（同一样本、同 KL 口径）：
A. input=actual_ids（旧喂法，buggy）→ 复现 loss≈14~24 / hit≈0.05
B. input=enc 全序列（含 BOS，修复）→ 学生=教师，loss 应≈0、hit≈0.9+
C. STE 包装 + skip-fake-quant（修复输入）→ 应与 B 一致（包装器正确性）
D. STE 包装 + fake-quant（修复输入）→ 真实 int4 退化基线

用法（容器内，单卡）：python probe_teacher_align.py [--max-len 512]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/yult/llm_infer_workspace/ylt/npuslim/src")
sys.path.insert(0, "/home/yult/llm_infer_workspace/ylt/qwen3-qad/precompute")

from npuslim.qad.teacher_cache import TeacherCache, FetchedSample  # noqa: E402
from npuslim.qad.kl_loss import kl_distill_loss    # noqa: E402
from npuslim.qad.ste_experts import wrap_all_experts  # noqa: E402
import npuslim.qad.ste_experts as _ste_exp          # noqa: E402
from precompute_logits import encode_one            # noqa: E402

MODEL_DIR = "/home/yult/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507"
CACHE = Path("/home/yult/llm_infer_workspace/ylt/qwen3-qad/precompute/logits_cache_topk100")
TRAIN = Path("/home/yult/llm_infer_workspace/ylt/glm52-qad/qad_data/qad_train_v1.jsonl")
CORPUS = Path("/home/yult/llm_infer_workspace/ylt/glm52-qad/glm52-qad-corpus/"
              "processed/corpus_v1_2026-08-27/clean")
VOCAB = 151_936
VCHUNK = 65536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    torch.npu.set_device(0)
    tc = TeacherCache(CACHE)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # ---- 取第一个可用样本（与 ds_train_t7 完全同口径） ----
    line_cache: dict[str, list[str]] = {}
    fs0 = None
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s["category"] not in ("agentic_coding", "general"):
                continue
            fname, _, ln = s["sample_key"].rpartition(":")
            if fname not in line_cache:
                with open(CORPUS / fname, encoding="utf-8") as fh:
                    line_cache[fname] = fh.readlines()
            row = json.loads(line_cache[fname][int(ln)].strip())
            enc = encode_one(tok, row, 8192)
            if not (enc is not None and not isinstance(enc, Exception) and len(enc) > 1):
                continue
            expected = np.asarray(enc[1:], dtype=np.int32)
            if expected.shape[0] < 20:
                continue
            fs = tc.fetch(s["id"], expected_ids=expected)
            L = min(fs.actual_ids.shape[0], args.max_len)
            fs0 = FetchedSample(sample_id=fs.sample_id, actual_ids=fs.actual_ids[:L],
                                topk_ids=fs.topk_ids[:L], topk_lps=fs.topk_lps[:L])
            break
    assert fs0 is not None, "no sample"
    # 修复输入：完整 token 序列（含 BOS）。teacher 行 i = P(enc[i+1] | enc[0..i])，
    # 学生输入含 enc[0..i] 时其 logits 行 i 与之严格对齐；可用行数 = min(教师行数, L_enc-1)
    L_enc = min(len(enc), args.max_len)
    input_np = np.asarray(enc, dtype=np.int64)[:L_enc]
    L_use = min(fs0.topk_ids.shape[0], L_enc - 1)
    assert int(fs0.actual_ids[0]) == int(enc[1]), "缓存 col0 与重编码流不匹配！"
    print(f"sample: {fs0.sample_id} enc_len={len(enc)} L_enc={L_enc} L_use={L_use} "
          f"(actual_ids[0]==enc[1] 校验通过)", flush=True)

    # ---- 加载（单卡 60GB，t6/t7 已验证可容纳） ----
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval().to("npu:0")
    print(f"load {time.time()-t0:.0f}s", flush=True)

    inp_bug = torch.from_numpy(fs0.actual_ids.astype(np.int64)).unsqueeze(0).to("npu:0")
    ids_bug = torch.from_numpy(fs0.topk_ids).long().to("npu:0")
    lps_bug = torch.from_numpy(fs0.topk_lps).float().to("npu:0")
    inp_fix = torch.from_numpy(input_np).unsqueeze(0).to("npu:0")
    ids_fix = torch.from_numpy(fs0.topk_ids[:L_use]).long().to("npu:0")
    lps_fix = torch.from_numpy(fs0.topk_lps[:L_use]).float().to("npu:0")

    def run_loss(tag, inp, ids_t, lps_t):
        with torch.no_grad():
            out = model(input_ids=inp, use_cache=False)
        lg = out.logits[0].float()
        L = min(lg.shape[0], ids_t.shape[0])   # 行 i 均预测 token i+1，行数取齐
        lg = lg[:L]
        loss = kl_distill_loss(lg, ids_t[:L], lps_t[:L], VOCAB, tau=1.0, chunk=VCHUNK)
        # 诊断：teacher top-1 vs 学生 argmax 命中率（前 128 token）
        n = min(128, L)
        hit = (lg[:n].argmax(-1) == ids_t[:n, 0]).float().mean()
        print(f"  [{tag}] loss={float(loss):.4f} top1_hit@{n}={float(hit):.3f} "
              f"mem={torch.npu.memory_allocated(0)/2**30:.1f}GiB", flush=True)
        del out, lg
        torch.npu.empty_cache()
        return float(loss)

    # A. 旧喂法（buggy）：input=actual_ids=enc[1:] → 学生预测 enc[i+2] vs 教师 enc[i+1]
    run_loss("A input=actual_ids (buggy)", inp_bug, ids_bug, lps_bug)
    # B. 修复：input=enc 全序列（含 BOS）→ 学生 logits 行 i == 教师行 i
    run_loss("B input=enc_full (fixed) plain", inp_fix, ids_fix, lps_fix)
    # C. STE 包装 + skip-fake-quant（修复输入）：应与 B 一致（包装器正确性）
    wrapped = wrap_all_experts(model, rank=args.rank, vectorized=True,
                               dtype=torch.bfloat16, use_checkpoint=False)
    for p in model.parameters():
        p.requires_grad = False
    _ste_exp.fake_quant_weight = lambda w, bits=4, scale=None: w
    run_loss("C ste+skip-fq (fixed)", inp_fix, ids_fix, lps_fix)
    # D. 恢复 fake-quant：真实 int4 退化基线
    from npuslim.qad.ste import fake_quant_weight as real_fq
    _ste_exp.fake_quant_weight = real_fq
    run_loss("D ste+fake-quant (fixed)", inp_fix, ids_fix, lps_fix)


if __name__ == "__main__":
    main()
