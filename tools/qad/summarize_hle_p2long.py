#!/usr/bin/env python3
"""Summarize HLE-MC P2-long (cap 16384) vs P2-8K on the 10 common questions."""
import json
import pandas as pd

D = "/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009"
meta = pd.read_parquet("/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_calib20.parquet")
subset = dict(zip(meta.id.astype(str), meta.subset))

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return {str(r["id"]): r for r in rows}

long = load(f"{D}/hle_mc_calib_p2long_l0.jsonl")
short = load(f"{D}/hle_mc_calib_p2_l0.jsonl")

def stats(d):
    n = len(d)
    return {
        "n": n,
        "correct": sum(r["correct"] for r in d.values()),
        "parsed": sum(r["parsed"] for r in d.values()),
        "trunc": sum(1 for r in d.values() if r.get("finish_reason") == "length"),
        "mean_tok": sum(r.get("usage", {}).get("completion_tokens", 0) for r in d.values()) / max(n, 1),
        "mean_s": sum(r.get("elapsed_s", 0) for r in d.values()) / max(n, 1),
        "max_s": max((r.get("elapsed_s", 0) for r in d.values()), default=0),
    }

print("== P2-long overall ==")
for k, v in stats(long).items():
    print(f"  {k}: {v}")
for ss in ("smoke10", "new10"):
    ids = [i for i in long if subset.get(i) == ss]
    st = stats({i: long[i] for i in ids})
    print(f"== P2-long {ss} (n={len(ids)}) ==")
    print("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))

common = sorted(set(long) & set(short))
print(f"== P2-long vs P2-8K on {len(common)} common questions ==")
ll = {i: long[i] for i in common}
ss = {i: short[i] for i in common}
a, b = stats(ll), stats(ss)
print(f"  P2-long: acc={a['correct']}/{a['n']}, parsed={a['parsed']}/{a['n']}, trunc={a['trunc']}, tok={a['mean_tok']:.0f}, s={a['mean_s']:.1f}")
print(f"  P2-8K  : acc={b['correct']}/{b['n']}, parsed={b['parsed']}/{b['n']}, trunc={b['trunc']}, tok={b['mean_tok']:.0f}, s={b['mean_s']:.1f}")
both = sum(1 for i in common if ll[i]["correct"] and ss[i]["correct"])
only_long = sum(1 for i in common if ll[i]["correct"] and not ss[i]["correct"])
only_short = sum(1 for i in common if not ll[i]["correct"] and ss[i]["correct"])
print(f"  both={both}, only_long={only_long}, only_8k={only_short}")
print("  per-question:")
for i in common:
    print(f"    {i[:8]} {subset.get(i,'?'):8s} long={ll[i]['pred']}/{ll[i]['correct']} "
          f"8k={ss[i]['pred']}/{ss[i]['correct']} gold={ll[i]['answer']} "
          f"tok={ll[i].get('usage',{}).get('completion_tokens')} "
          f"trunc={ll[i].get('finish_reason')}")
