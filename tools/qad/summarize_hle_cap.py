#!/usr/bin/env python3
"""Compare HLE-MC smoke10 at cap 16K / 24K / 32K (and 8K where available)."""
import json

D = "/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009"
paths = {
    "8K": f"{D}/hle_mc_calib_p2_l0.jsonl",
    "16K": f"{D}/hle_mc_calib_p2long_l0.jsonl",
    "24K": f"{D}/hle_mc_cap24576_l0.jsonl",
    "32K": f"{D}/hle_mc_cap32768_l0.jsonl",
}

def load(path):
    try:
        return {str(json.loads(l)["id"]): json.loads(l) for l in open(path)}
    except FileNotFoundError:
        return {}

def stats(d):
    n = len(d)
    return {
        "n": n,
        "acc": sum(r["correct"] for r in d.values()),
        "parsed": sum(r["parsed"] for r in d.values()),
        "trunc": sum(1 for r in d.values() if r.get("finish_reason") == "length"),
        "mean_tok": sum(r.get("usage", {}).get("completion_tokens", 0) for r in d.values()) / max(n, 1),
        "mean_s": sum(r.get("elapsed_s", 0) for r in d.values()) / max(n, 1),
        "max_s": max((r.get("elapsed_s", 0) for r in d.values()), default=0),
    }

data = {k: load(v) for k, v in paths.items()}
print("== cap ladder (smoke10; 16K file includes 20, filter to first 10 ids) ==")
# The 16K file is 20; keep the smoke10 subset only.
import pandas as pd
meta = pd.read_parquet("/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_calib20.parquet")
smoke_ids = set(meta[meta.subset == "smoke10"].id.astype(str))
if data["16K"]:
    data["16K"] = {i: r for i, r in data["16K"].items() if i in smoke_ids}
for k, d in data.items():
    if not d:
        print(f"== {k}: missing ==")
        continue
    s = stats(d)
    print(f"{k}: n={s['n']} acc={s['acc']}/{s['n']} parsed={s['parsed']}/{s['n']} "
          f"trunc={s['trunc']} tok={s['mean_tok']:.0f} s={s['mean_s']:.1f} max_s={s['max_s']:.1f}")
print("== per question ==")
ids = sorted(set().union(*[set(d) for d in data.values()]))
for i in ids:
    row = [i[:8]]
    for k in ("8K", "16K", "24K", "32K"):
        r = data.get(k, {}).get(i)
        if r:
            row.append(f"{k}:{r['pred']}/{r['correct']}/tok{r.get('usage',{}).get('completion_tokens')}/{'TR' if r.get('finish_reason')=='length' else 'ok'}")
        else:
            row.append(f"{k}:-")
    gold = next((r.get("answer", "?") for r in data.values() if i in r), "?")
    print(f"  {' | '.join(row)} | gold={gold}")
