#!/usr/bin/env python3
"""HLE text-only exact-match probe (local frontier ceiling axis).

Protocol (all arms): chat endpoint, thinking on (reasoning_effort=high),
temperature=0.6, max_tokens=32768. Final answer extracted from
"Exact Answer:" (fallback: last \\boxed{}, then last non-empty line) and
compared to the gold answer after NFKC/lower/whitespace normalization.

Input : hle_smoke20.parquet / hle_120.parquet
Output: jsonl + _summary.json (overall, by_category, parse/trunc, timing)

Usage:
  python3 hle_probe.py --api http://HOST:PORT/v1/chat/completions \
      --model front_l0 --data .../hle_smoke20.parquet \
      --out .../hle_smoke20_l0.jsonl --workers 2
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_EXACT_RE = re.compile(r"(?i)Exact Answer\s*:\s*(.+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(question):
    return (
        "Answer the following question. Think step by step, then provide your final "
        "answer in the format:\nExact Answer: <your answer>\n\n"
        f"Question:\n{question}"
    )


def clean(s):
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.strip(" \t\r\n`'\"")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def norm(s):
    return clean(s).lower()


def extract_answer(text):
    text = text or ""
    hits = _EXACT_RE.findall(text)
    if hits:
        return clean(hits[-1]), True
    hits = _BOXED_RE.findall(text)
    if hits:
        return clean(hits[-1]), True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return clean(lines[-1]), False
    return "", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit).copy()
    cases = []
    for _, r in df.iterrows():
        cases.append({
            "id": str(r["id"]),
            "category": str(r["category"]),
            "gold": str(r["answer"]),
            "prompt": build_prompt(str(r["question"])),
        })
    if args.dry_run:
        for c in cases[:2]:
            print("=" * 80)
            print(c["id"], c["category"], "gold=", c["gold"])
            print(c["prompt"][:1500])
        print(f"dry-run: {len(cases)} cases; first prompt chars={len(cases[0]['prompt'])}")
        return

    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    todo = [c for c in cases if c["id"] not in done]
    print(f"loaded {len(cases)} cases; already done {len(done)}; todo {len(todo)}", flush=True)

    def one(c):
        t0 = time.time()
        rec = {"id": c["id"], "category": c["category"], "gold": c["gold"]}
        try:
            payload = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {
                    "reasoning_effort": args.reasoning_effort,
                },
                "messages": [{"role": "user", "content": c["prompt"]}],
            }
            resp = post(args.api, payload)
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            pred, parsed = extract_answer(content)
            rec.update({
                "pred": pred,
                "parsed": parsed,
                "correct": int(parsed and norm(pred) == norm(c["gold"])),
                "contains": int(bool(pred) and norm(c["gold"]) in norm(pred)),
                "finish_reason": choice.get("finish_reason"),
                "ans": content.strip()[-800:],
                "usage": resp.get("usage", {}),
            })
        except Exception as e:
            rec.update({"pred": "", "parsed": False, "correct": 0,
                        "contains": 0, "error": repr(e)[:200]})
        rec["elapsed_s"] = round(time.time() - t0, 2)
        return rec

    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 5 == 0:
                print(f"  [{i}/{len(todo)}]", flush=True)

    rows = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    parsed = sum(r["parsed"] for r in rows)
    contains = sum(r["contains"] for r in rows)
    trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
    errors = sum(1 for r in rows if "error" in r)
    comp = [r.get("usage", {}).get("completion_tokens", 0) for r in rows]
    elapsed = [r.get("elapsed_s", 0.0) for r in rows]
    by_cat = {}
    for r in rows:
        b = by_cat.setdefault(r["category"], [0, 0])
        b[0] += r["correct"]; b[1] += 1
    summary = {
        "n": n,
        "overall": correct / max(n, 1),
        "parsed_rate": parsed / max(n, 1),
        "contains_rate": contains / max(n, 1),
        "truncated": trunc,
        "errors": errors,
        "mean_completion_tokens": sum(comp) / max(n, 1),
        "mean_elapsed_s": sum(elapsed) / max(n, 1),
        "by_category": {k: {"correct": v[0], "n": v[1], "acc": v[0] / max(v[1], 1)}
                        for k, v in sorted(by_cat.items())},
        "protocol": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "thinking": True,
        },
    }
    out_sum = args.out.replace(".jsonl", "_summary.json")
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== HLE: {correct}/{n} = {summary['overall']:.3f}; parsed={parsed}/{n}; "
          f"contains={contains}/{n}; trunc={trunc}; errors={errors}; "
          f"mean_tok={summary['mean_completion_tokens']:.0f}; mean_s={summary['mean_elapsed_s']:.1f}")
    for k, v in summary["by_category"].items():
        print(f"  {k}: {v['correct']}/{v['n']} = {v['acc']:.3f}")
    print(f"wrote {out_sum}")


if __name__ == "__main__":
    main()
