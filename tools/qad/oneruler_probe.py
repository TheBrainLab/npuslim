#!/usr/bin/env python3
"""OneRuler multilingual long-context probe (26-lang NIAH family; sampled subset).

Data: starbix/oneruler mirror (4K/8K/16K token contexts; 7 tasks:
niah_single / niah_none / niah_multikey / niah_multiquery / niah_multivalue / cwe / cwe_easy).
Sampling: LANGS x LENGTHS x tasks x PER_CELL cases (fixed seed).

Protocol (fixed, all arms): chat endpoint, enable_thinking=FALSE, temperature=0, max_tokens=64.
Scoring:
  * niah_single/multikey/multiquery/multivalue : ALL answer strings must appear (substring, case-insensitive)
  * niah_none                                  : response must contain "none" (no hallucinated numbers)
  * cwe / cwe_easy                             : recall = matched / len(answer); correct = all matched
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/frontier/oneruler"
LANGS = ["en", "zh", "hi", "sw", "st", "ta"]
LENGTHS = [4096, 8192, 16384]
TASKS = ["niah_single", "niah_none", "niah_multikey", "niah_multiquery", "niah_multivalue", "cwe", "cwe_easy"]
PER_CELL = 4

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    import pandas as pd
    rng = random.Random(args.seed)
    cases = []
    for lang in LANGS:
        for length in LENGTHS:
            p = f"{DATA_DIR}/{lang}_{length}.parquet"
            if not os.path.exists(p):
                print(f"  WARN missing {p}", flush=True)
                continue
            df = pd.read_parquet(p)
            for task in TASKS:
                sub = df[df["task"] == task]
                if len(sub) == 0:
                    continue
                take = sub.sample(n=min(PER_CELL, len(sub)), random_state=args.seed)
                for _, r in take.iterrows():
                    cases.append({
                        "lang": lang, "length": length, "task": task,
                        "prompt": str(r["context"]) + "\n\n" + str(r["question"]) + "\n\n" + str(r["answer_prefix"]),
                        "answers": [str(a) for a in r["answer"]],
                    })
    print(f"loaded {len(cases)} OneRuler cases", flush=True)

    def one(c):
        rec = {k: c[k] for k in ("lang", "length", "task")}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 64,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": c["prompt"]}]})
            ans = (resp["choices"][0]["message"].get("content") or "").lower()
            rec["ans"] = ans.strip()[:150]
            if c["task"] == "niah_none":
                rec["hits"], rec["total"], rec["correct"] = (1 if "none" in ans else 0), 1, int("none" in ans)
            else:
                hits = sum(1 for a in c["answers"] if a.lower() in ans)
                rec["hits"], rec["total"] = hits, len(c["answers"])
                rec["correct"] = int(hits == len(c["answers"]))
        except Exception as e:
            rec.update({"correct": 0, "hits": 0, "total": len(c["answers"]), "error": repr(e)[:150]})
        return rec

    res = []
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(one, c) for c in cases]):
            r = fut.result()
            res.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if len(res) % 50 == 0:
                print(f"  [{len(res)}/{len(cases)}]", flush=True)

    summary = {"n": len(res), "by_task": {}, "by_lang": {}, "by_length": {}, "by_task_lang": {}}
    for r in res:
        for key, bucket in (("task", "by_task"), ("lang", "by_lang"), ("length", "by_length")):
            b = summary[bucket].setdefault(r[key], [0, 0])
            b[0] += r["correct"]; b[1] += 1
        b = summary["by_task_lang"].setdefault(f"{r['task']}|{r['lang']}", [0, 0])
        b[0] += r["correct"]; b[1] += 1
    n = max(len(res), 1)
    overall = sum(r["correct"] for r in res) / n
    print(f"=== OneRuler summary: {sum(r['correct'] for r in res)}/{n} = {overall:.3f}")
    for bucket in ("by_task", "by_lang", "by_length"):
        print(f"  {bucket}:")
        for k, (ok, tot) in sorted(summary[bucket].items()):
            print(f"    {k}: {ok}/{tot} = {ok/tot:.3f}")
    summary["overall"] = overall
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
