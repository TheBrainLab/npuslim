#!/usr/bin/env python3
"""SimpleQA probe (calibration / abstention axis).

Protocol (fixed): chat endpoint, enable_thinking=false, temperature=0, max_tokens=256.
Metrics:
  * attempted rate  : response is NOT an abstention ("I don't know" / refusal family)
  * correct rate    : approximate fuzzy match of the gold answer in the response
                      (the official SimpleQA uses a GPT-4 judge; this is a
                       token-overlap approximation — comparable ACROSS products,
                       not to published numbers)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

ABSTAIN = [re.compile(p, re.I) for p in [
    r"\bI don'?t know\b", r"\bI do not know\b", r"\bI'?m not sure\b", r"\bI am not sure\b",
    r"\bI cannot (answer|determine|say)\b", r"\bI can'?t (answer|determine|say)\b",
    r"\bunknown\b", r"\bunable to (answer|determine)\b", r"\bI'?m not able\b",
    r"我不知道", r"无法回答", r"不清楚",
]]


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(" " if c in string.punctuation else c for c in s)
    return " ".join(s.split())


def fuzzy_correct(gold: str, resp: str) -> bool:
    g = norm(gold)
    r = norm(resp)
    if not g:
        return False
    if g in r:
        return True
    gt = [t for t in g.split() if len(t) > 2]
    if not gt:
        return g in r
    hit = sum(1 for t in gt if re.search(r"\b" + re.escape(t) + r"\b", r))
    return hit / len(gt) >= 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(f"{args.data_dir}/simpleqa_test.csv")))
    if args.limit:
        rows = rows[:args.limit]

    def one(row):
        q = row["problem"]
        gold = row["answer"]
        rec = {"problem": q[:200], "gold": gold}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 1024,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": q}]})
            content = resp["choices"][0]["message"]["content"] or ""
            abstained = any(p.search(content[:400]) for p in ABSTAIN)
            rec.update({"abstained": int(abstained),
                        "correct": int((not abstained) and fuzzy_correct(gold, content)),
                        "content": content[:300]})
        except Exception as e:
            rec.update({"error": repr(e)[:200], "abstained": -1, "correct": 0})
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = attempted = correct = err = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            n += 1
            if rec.get("abstained", -1) >= 0:
                attempted += int(rec["abstained"] == 0)
                correct += rec.get("correct", 0)
            else:
                err += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 200 == 0:
                print(f"  [{n}/{len(rows)}]", flush=True)

    print("=== summary ===")
    print(f"  attempted rate = {attempted}/{n} = {attempted/max(n,1):.4f}")
    print(f"  correct (approx) = {correct}/{n} = {correct/max(n,1):.4f} | correct|attempted = {correct/max(attempted,1):.4f} | err={err}")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({"n": n, "attempted": attempted, "correct": correct, "err": err}, f, indent=1)


if __name__ == "__main__":
    main()
