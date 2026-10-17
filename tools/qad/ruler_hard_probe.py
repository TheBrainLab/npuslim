#!/usr/bin/env python3
"""RULER-style HARD long-context probe (the plain NIAH hit a ceiling).

Three harder sub-tasks at a fixed length (default 64K, TP=16-consistent):
  * multi_key8 : 8 distinct keys, one needle each; ask for ONE key's value
  * multi_value: ONE key with 4 values; ask for ALL of its values
  * var_track  : a chain of ~40 arithmetic assignments; ask for the FINAL value

Protocol (fixed): chat endpoint, enable_thinking=false, temperature=0, max_tokens=64.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request

WIKITEXT = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/wikitext_train.txt"
CHARS_PER_TOKEN = 4.0

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_haystack() -> str:
    return open(WIKITEXT).read()


def build(hay: str, n_chars: int, inserts: list[tuple[float, str]], query: str) -> str:
    body = hay[:n_chars]
    out, prev = [], 0
    for frac, snip in sorted(inserts, key=lambda x: x[0]):
        pos = max(prev + 1, min(int(frac * len(body)), len(body)))
        out.append(body[prev:pos])
        out.append("\n" + snip + "\n")
        prev = pos
    out.append(body[prev:])
    return "".join(out) + f"\n\nQuestion: {query}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--per-task", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hay = load_haystack()
    n_chars = min(int(args.length * CHARS_PER_TOKEN), len(hay))
    cases = []

    # (a) multi-key 8
    for i in range(args.per_task):
        keys = [f"K{rng.randint(1000, 9999)}" for _ in range(8)]
        vals = {k: str(rng.randint(100000, 999999)) for k in keys}
        ask = rng.choice(keys)
        inserts = [(rng.uniform(0.05, 0.95), f"One of the special magic numbers for {k} is: {v}.") for k, v in vals.items()]
        q = f"What is the special magic number for {ask} mentioned in the provided text?"
        cases.append({"task": "multi_key8", "gold": vals[ask], "prompt": build(hay, n_chars, inserts, q)})

    # (b) multi-value (one key, 4 values) -> ask for all
    for i in range(args.per_task):
        key = f"K{rng.randint(1000, 9999)}"
        vals = [str(rng.randint(100000, 999999)) for _ in range(4)]
        inserts = [(rng.uniform(0.05, 0.95), f"One of the special magic numbers for {key} is: {v}.") for v in vals]
        q = f"What are ALL the special magic numbers for {key} mentioned in the provided text? List them."
        cases.append({"task": "multi_value", "gold": vals, "prompt": build(hay, n_chars, inserts, q)})

    # (c) variable tracking (~40 steps)
    for i in range(args.per_task):
        names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        cur = rng.randint(1, 20)
        steps = []
        for j in range(40):
            nxt = rng.choice(names)
            op = rng.choice(["+", "-"])
            k = rng.randint(2, 9)
            cur = cur + k if op == "+" else cur - k
            steps.append(f"{nxt} = {cur}.")
        inserts = [(0.05 + 0.9 * (j / len(steps)), s) for j, s in enumerate(steps)]
        last_var = steps[-1].split(" ")[0]
        q = (f"In the provided text there is a sequence of assignments; "
             f"what is the FINAL value assigned to the variable {last_var}?")
        cases.append({"task": "var_track", "gold": str(cur), "prompt": build(hay, n_chars, inserts, q)})

    def one(rec):
        out = {k: v for k, v in rec.items() if k != "prompt"}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 64,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": rec["prompt"]}]})
            ans = resp["choices"][0]["message"]["content"] or ""
            if rec["task"] == "multi_value":
                hit = sum(1 for v in rec["gold"] if v in ans)
                out.update({"hits": hit, "total": len(rec["gold"]), "correct": int(hit == len(rec["gold"]))})
            else:
                out.update({"correct": int(str(rec["gold"]) in ans)})
            out["answer"] = ans.strip()[:150]
        except Exception as e:
            out.update({"correct": 0, "error": repr(e)[:200]})
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    summary = {}
    n_done = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in cases]
        for fut in as_completed(futs):
            r = fut.result()
            s = summary.setdefault(r["task"], {"n": 0, "correct": 0, "hits": 0, "total": 0, "err": 0})
            s["n"] += 1
            s["correct"] += r.get("correct", 0)
            if r["task"] == "multi_value":
                s["hits"] += r.get("hits", 0)
                s["total"] += r.get("total", 0)
            s["err"] += int(bool(r.get("error")))
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            print(f"  [{n_done}/{len(cases)}] {r['task']} correct={r.get('correct')}", flush=True)

    print("=== summary (L={}) ===".format(args.length))
    for t, s in sorted(summary.items()):
        extra = f" | value-recall {s['hits']}/{s['total']}" if t == "multi_value" else ""
        print(f"  {t}: {s['correct']}/{s['n']} = {s['correct']/max(s['n'],1):.3f}{extra} (err={s['err']})")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
