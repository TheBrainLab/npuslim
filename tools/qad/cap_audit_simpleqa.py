#!/usr/bin/env python3
"""Cap audit: re-ask the first N SimpleQA rows with a large cap (1024) and compare
fuzzy correctness against the recorded run (cap 256). Detects 坑61-class contamination.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd")
from simpleqa_probe import fuzzy_correct  # noqa: E402

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recorded", required=True, help="the original simpleqa jsonl to compare")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open("/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/simpleqa_test.csv")))[:args.limit]
    rec = {}
    for line in open(args.recorded):
        r = json.loads(line)
        rec[r["problem"][:200]] = r

    def one(row):
        q, gold = row["problem"], row["answer"]
        out = {"problem": q[:200], "gold": gold}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": args.cap,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": q}]})
            ch = resp["choices"][0]
            content = ch["message"]["content"] or ""
            out.update({"correct_1024": int(fuzzy_correct(gold, content)),
                        "finish": ch.get("finish_reason"),
                        "len": len(content), "tail": content.strip()[-120:]})
        except Exception as e:
            out.update({"correct_1024": 0, "error": repr(e)[:150]})
        return out

    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in rows]):
            res.append(fut.result())

    agree = changed_up = changed_down = missing = 0
    details = []
    for r in res:
        old = rec.get(r["problem"])
        if old is None:
            missing += 1
            continue
        o, n = int(old["correct"]), r["correct_1024"]
        if o == n:
            agree += 1
        elif n > o:
            changed_up += 1
        else:
            changed_down += 1
        details.append({"problem": r["problem"][:60], "gold": r["gold"][:40],
                        "old_correct": o, "new_correct": n, "finish": r.get("finish"),
                        "len": r.get("len"), "tail": r.get("tail")})

    summary = {"n": len(res), "agree": agree, "up": changed_up, "down": changed_down, "missing": missing,
               "cap": args.cap, "finish_length": sum(1 for r in res if r.get("finish") == "length")}
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
