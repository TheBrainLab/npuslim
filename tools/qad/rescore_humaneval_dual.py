#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""humaneval 离线 fence-extract 口径重评（1011，★ 参考/非主判据）。

⚠️ 与 audit 主档的 "trunc" 口径不同（后者 = 幻觉标记处截断，见
tools/deploy/reeval_humaneval_trunc.py；audit 主口径 = trunc）。本口径在
W8 样本上仅 ~0.15（围栏内代码体执行失败多），不可与 trunc 0.93+ 直接对比，
仅供提取链诊断。协议：
  prediction = doc["prompt"] + fence_extract(completion)
  reference = doc["test"] + "\n" + "check(" + doc["entry_point"] + ")"
纯 CPU 无 NPU。用法（容器内）：
  python3 rescore_humaneval_dual.py <samples.jsonl> <out.json>
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
import evaluate as hf_evaluate

CODE_EVAL = "/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8/code_eval.py"


def extract_code_blocks(text: str) -> str:
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    matches = re.findall(pattern, "```" + text, re.DOTALL)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        return ""
    return matches[0]


def main():
    samples_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    recs = [json.loads(l) for l in open(samples_file)]
    n = len(recs)
    refs = [r["doc"]["test"] + "\ncheck(" + r["doc"]["entry_point"] + ")" for r in recs]
    predictions_raw = [[r["doc"]["prompt"] + r["resps"][0][0]] for r in recs]
    predictions_trunc = [[r["doc"]["prompt"] + extract_code_blocks(r["resps"][0][0])] for r in recs]

    compute_ = hf_evaluate.load(CODE_EVAL)
    out = {"n": n, "file": os.path.basename(samples_file)}

    for tag, preds in [("raw", predictions_raw), ("fence", predictions_trunc)]:
        pass_at_k, results = compute_._compute(preds, refs, k=[1], num_workers=16, timeout=3.0)
        per_sample = {}
        for i in range(n):
            per_sample[recs[i]["doc"]["task_id"]] = bool(results[i][0][1]["passed"])
        out[tag] = {
            "pass_at_1": float(pass_at_k["pass@1"]),
            "passed": int(sum(per_sample.values())),
        }
        fails = sorted(t for t, ok in per_sample.items() if not ok)
        out[tag]["failed_task_ids"] = fails
        print(f"{tag}: n={n} pass@1 = {out[tag]['pass_at_1']:.4f} (passed={out[tag]['passed']})")
    if out_file:
        with open(out_file, "w") as f:
            json.dump(out, f, indent=1)
        print("saved ->", out_file)


if __name__ == "__main__":
    main()
