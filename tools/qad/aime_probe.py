#!/usr/bin/env python3
"""AIME 2026 probe (MathArena/aime_2026, 30 problems) -- align with the msmodelslim w4a8c8 card.

Protocol (fixed, all arms): chat endpoint, enable_thinking=TRUE, temperature=0,
max_tokens=16384 (thinking budget, equal for all arms).
Prompt: problem + "End your response with: Exact Answer: <integer>".
Scoring: exact match of the extracted integer vs the official answer.
Records: idx, gold, pred, correct, n_out_tok (thinking budget used), finish.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/frontier/aime_2026.parquet"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

ANS_RE = re.compile(r"Exact Answer:\s*\**\s*(-?\d{1,6})", re.I)
BOX_RE = re.compile(r"\\?boxed\{\s*(-?\d{1,6})\s*\}")


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract(ans: str):
    m = ANS_RE.findall(ans)
    if m:
        return m[-1]
    m = BOX_RE.findall(ans)
    if m:
        return m[-1]
    m = re.findall(r"(?<![\d.])(\d{1,3})(?![\d.])", ans)  # last small integer fallback
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(DATA)
    problems = [{"idx": int(r["problem_idx"]), "problem": str(r["problem"]), "gold": str(int(r["answer"]))}
                for _, r in df.iterrows()]
    print(f"loaded {len(problems)} AIME 2026 problems", flush=True)

    def one(p):
        rec = {"idx": p["idx"], "gold": p["gold"]}
        prompt = p["problem"] + "\n\nSolve the problem. End your response with a line in the format: Exact Answer: <integer>"
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                                   "chat_template_kwargs": {"enable_thinking": True},
                                   "messages": [{"role": "user", "content": prompt}]})
            ch = resp["choices"][0]
            content = ch["message"].get("content") or ""
            reasoning = ch["message"].get("reasoning_content") or ""
            full = reasoning + "\n" + content
            pred = extract(full)
            rec.update({"pred": pred, "correct": int(pred == p["gold"]),
                        "finish": ch.get("finish_reason"),
                        "n_out_tok": ch.get("usage", {}).get("completion_tokens"),
                        "tail": content.strip()[-120:]})
        except Exception as e:
            rec.update({"correct": 0, "error": repr(e)[:200]})
        return rec

    res = []
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(one, p) for p in problems]):
            r = fut.result()
            res.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  [{len(res)}/{len(problems)}] correct={r.get('correct')} gold={r.get('gold')} pred={r.get('pred')}", flush=True)

    n = len(res)
    correct = sum(r.get("correct", 0) for r in res)
    toks = [r.get("n_out_tok") for r in res if r.get("n_out_tok")]
    summary = {"n": n, "correct": correct, "acc": correct / max(n, 1),
               "mean_out_tok": sum(toks) / max(len(toks), 1),
               "trunc": sum(1 for r in res if r.get("finish") == "length"),
               "err": sum(1 for r in res if r.get("error"))}
    print(f"=== AIME 2026 summary: {correct}/{n} = {summary['acc']:.3f} "
          f"(mean out tok {summary['mean_out_tok']:.0f}, trunc {summary['trunc']}, err {summary['err']})")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
