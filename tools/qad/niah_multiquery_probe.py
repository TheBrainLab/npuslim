#!/usr/bin/env python3
"""Local RULER-style multiquery NIAH at 64K/128K.

Design:
  * haystack = wikitext-2 train text (~4 chars/token)
  * insert qsize needles (keys alpha..theta) at evenly spaced depths
  * ask for ALL qsize values in one question
  * score: all values present (correct) + mean recall (hits/qsize)

Cells: lengths {65536,131072} x qsize {2,4,8} x per_cell cases.
Protocol: non-thinking, temp 0, max_tokens 256, workers 2.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

WIKITEXT = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/wikitext_train.txt"
CHARS_PER_TOKEN = 4.0
KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_haystack(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    if len(s) < 600000:
        raise SystemExit(f"haystack too short: {len(s)} chars")
    return s


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_case(hay, n_chars, needles, depths, keys):
    body = hay[:n_chars]
    out, prev = [], 0
    for (key, val), d in sorted(zip(needles, depths), key=lambda x: x[1]):
        pos = int(d * len(body))
        if pos <= prev:
            pos = min(prev + 1, len(body))
        out.append(body[prev:pos])
        out.append(f"\nOne of the special magic numbers for {key} is: {val}.\n")
        prev = pos
    out.append(body[prev:])
    text = "".join(out)
    key_list = ", ".join(keys)
    q = (
        f"What are the special magic numbers for the following keys mentioned in "
        f"the provided text: {key_list}? List all of them in the format 'key: value'."
    )
    return f"{text}\n\nQuestion: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="65536,131072")
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hay = load_haystack(WIKITEXT)
    cases = []
    for L in [int(x) for x in args.lengths.split(",")]:
        n_chars = min(int(L * CHARS_PER_TOKEN), len(hay))
        for qsize in (2, 4, 8):
            keys = KEYS[:qsize]
            for i in range(args.per_cell):
                vals = {k: str(rng.randint(100000, 999999)) for k in keys}
                depths = [((j + 0.5) / qsize) for j in range(qsize)]
                prompt = build_case(hay, n_chars, list(vals.items()), depths, keys)
                cases.append({
                    "length": L, "qsize": qsize, "keys": keys,
                    "answers": [vals[k] for k in keys], "prompt": prompt,
                })
    print(f"loaded {len(cases)} multiquery cases; first prompt chars={len(cases[0]['prompt'])}", flush=True)
    if args.dry_run:
        for c in cases[:2]:
            print("=" * 80)
            print(c["length"], "qsize", c["qsize"], "answers", c["answers"])
            print(c["prompt"][:1000])
        return

    def one(c):
        rec = {k: c[k] for k in ("length", "qsize", "keys", "answers")}
        try:
            resp = post(args.api, {
                "model": args.model, "temperature": 0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": c["prompt"]}],
            })
            ans = (resp["choices"][0]["message"].get("content") or "").lower()
            hits = sum(1 for a in c["answers"] if a.lower() in ans)
            rec.update({
                "hits": hits, "total": len(c["answers"]),
                "recall": hits / len(c["answers"]),
                "correct": int(hits == len(c["answers"])),
                "ans": ans.strip()[:300],
            })
        except Exception as e:
            rec.update({"hits": 0, "total": len(c["answers"]), "recall": 0.0,
                        "correct": 0, "error": repr(e)[:200]})
        return rec

    res = []
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, c) for c in cases]), 1):
            r = fut.result()
            res.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0:
                print(f"  [{i}/{len(cases)}]", flush=True)

    summary = {"n": len(res), "by_cell": {}}
    for r in res:
        k = f"L{r['length']}_q{r['qsize']}"
        b = summary["by_cell"].setdefault(k, {"correct": 0, "n": 0, "recall_sum": 0.0})
        b["correct"] += r["correct"]; b["n"] += 1; b["recall_sum"] += r["recall"]
    for k, b in summary["by_cell"].items():
        b["acc"] = b["correct"] / max(b["n"], 1)
        b["mean_recall"] = b["recall_sum"] / max(b["n"], 1)
    summary["overall_correct"] = sum(r["correct"] for r in res) / max(len(res), 1)
    summary["overall_recall"] = sum(r["recall"] for r in res) / max(len(res), 1)
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("=== multiquery summary ===")
    for k, b in sorted(summary["by_cell"].items()):
        print(f"  {k}: acc={b['correct']}/{b['n']} recall={b['mean_recall']:.3f}")
    print(f"  overall: acc={summary['overall_correct']:.3f} recall={summary['overall_recall']:.3f}")


if __name__ == "__main__":
    main()
