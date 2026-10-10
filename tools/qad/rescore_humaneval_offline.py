#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线重判 W4A8 基线 humaneval samples。
原 run（tier1_base_r2_he）中 create_test 未执行（utils 导入时 hf_evaluate.load("code_eval")
离线失败 → result 全 None，报出的 pass@1 0.2622 无效）。
协议严格对齐 lm_eval 0.4.12 create_test：
  prediction = doc["prompt"] + completion（build_predictions）
  reference  = doc["test"] + "\n" + "check(" + doc["entry_point"] + ")"（doc_to_target）
  本地 code_eval.py _compute(num_workers, timeout=3.0 默认)
纯 CPU，无 NPU。用法（容器内）:
  python3 rescore_humaneval_offline.py <samples.jsonl>
"""
import json
import os
import sys

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
import evaluate as hf_evaluate

CODE_EVAL = "/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8/code_eval.py"


def main():
    samples_file = sys.argv[1]
    recs = [json.loads(l) for l in open(samples_file)]
    n = len(recs)
    predictions, references = [], []
    for r in recs:
        d = r["doc"]
        comp = r["resps"][0][0]
        predictions.append([d["prompt"] + comp])
        references.append(d["test"] + "\ncheck(" + d["entry_point"] + ")")
    compute_ = hf_evaluate.load(CODE_EVAL)
    pass_at_k, results = compute_._compute(predictions, references, k=[1], num_workers=16, timeout=3.0)
    per_sample = {}
    passed = 0
    for i in range(n):
        ok = results[i][0][1]["passed"]
        per_sample[recs[i]["doc"]["task_id"]] = ok
        passed += int(ok)
    print(f"n={n}")
    print(f"pass@1 = {pass_at_k['pass@1']:.4f}  (passed={passed})")
    fails = sorted([t for t, ok in per_sample.items() if not ok])
    print(f"failed task_ids ({len(fails)}): {fails}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump({"pass_at_1": float(pass_at_k["pass@1"]), "passed": passed, "n": n,
                       "per_sample": per_sample}, f, indent=1)
        print(f"saved -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
