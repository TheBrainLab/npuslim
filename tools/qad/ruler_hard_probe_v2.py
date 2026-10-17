#!/usr/bin/env python3
"""RULER-hard v2 (1017 fix): GENUINE variable tracking.

v1 defect: each assignment wrote a PRE-COMPUTED value ("beta = 28.") -> the task
degenerated into "find the last assignment of <name>" (retrieval), not tracking.
v2 fix  : assignments CHAIN references ("beta = alpha + 3.") -> the model must
resolve the whole chain (40 steps, 8 recycled names) and compute the final value.

Protocol (fixed): chat endpoint, enable_thinking=false, temperature=0, max_tokens=64.
Scoring: word-boundary number match (gold is a small int; substring match would
false-positive, e.g. "12" inside "112").
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


def num_in(ans: str, gold: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(gold)}(?!\d)", ans) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=65536)
    ap.add_argument("--per-task", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hay = load_haystack()
    n_chars = min(int(args.length * CHARS_PER_TOKEN), len(hay))
    cases = []

    for i in range(args.per_task):
        names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        cur_name = rng.choice(names)
        cur_val = rng.randint(1, 20)
        steps = [f"{cur_name} = {cur_val}."]
        for j in range(40):
            nxt = rng.choice([n for n in names if n != cur_name])
            op = rng.choice(["+", "-"])
            k = rng.randint(2, 9)
            steps.append(f"{nxt} = {cur_name} {op} {k}.")
            cur_val = cur_val + k if op == "+" else cur_val - k
            cur_name = nxt
        inserts = [(0.05 + 0.9 * (j / len(steps)), s) for j, s in enumerate(steps)]
        q = (f"In the provided text there is a sequence of assignments; "
             f"what is the FINAL value assigned to the variable {cur_name}?")
        cases.append({"task": "var_track_chain", "gold": str(cur_val),
                      "prompt": build(hay, n_chars, inserts, q)})

    def one(rec):
        out = {k: v for k, v in rec.items() if k != "prompt"}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 64,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": rec["prompt"]}]})
            ans = resp["choices"][0]["message"]["content"] or ""
            out.update({"correct": int(num_in(ans, rec["gold"])), "answer": (lambda a: a if len(a) <= 2000 else a[:600] + " … " + a[-1400:])(ans.strip())})
        except Exception as e:
            out.update({"correct": 0, "error": repr(e)[:200]})
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    summary = {"n": 0, "correct": 0, "err": 0}
    n_done = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in cases]
        for fut in as_completed(futs):
            r = fut.result()
            summary["n"] += 1
            summary["correct"] += r.get("correct", 0)
            summary["err"] += int(bool(r.get("error")))
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            print(f"  [{n_done}/{len(cases)}] var_track_chain correct={r.get('correct')} "
                  f"gold={r.get('gold')} ans={r.get('answer','')[:60]}", flush=True)

    n = max(summary["n"], 1)
    print(f"=== summary (L={args.length}) ===\n  var_track_chain: "
          f"{summary['correct']}/{summary['n']} = {summary['correct']/n:.3f} (err={summary['err']})")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
