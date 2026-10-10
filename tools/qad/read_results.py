#!/usr/bin/env python3
"""Read eval results and print mmlu/gsm8k."""
import json, glob
for f in sorted(glob.glob("/llm_infer_workspace/ylt/glm52-qad/eval_run14_step600/*/results_*.json")):
    d = json.load(open(f))
    print("===", f.split("/")[-1], "===")
    for t, r in sorted(d.get("results", {}).items()):
        for k, v in sorted(r.items()):
            if isinstance(v, (int, float)) and any(s in k for s in ["acc", "exact", "match", "pass", "ppl", "strict"]):
                print(f"  {t} {k} = {round(v, 6)}")
