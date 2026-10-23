#!/usr/bin/env python3
"""HLE multiple-choice probe (text-only; letter exact-match).

Input question text already contains "Answer Choices: ...". Protocol:
thinking on (reasoning_effort=high), temperature=0.6, max_tokens=16384.
Extraction: 'ANSWER: X' (fallback: last standalone capital letter on the
last non-empty line). Score: letter exact-match.

Usage:
  python3 hle_mc_probe.py --api http://HOST:PORT/v1/chat/completions \
      --model hle_l0 --data .../hle_mc_smoke10.parquet \
      --out .../hle_mc_smoke10_l0.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_ANS_RE = re.compile(r"(?i)ANSWER\s*:\s*([A-Z])")
_LAST_RE = re.compile(r"\b([A-Z])\b")


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(question, direct=False):
    if direct:
        return (
            "Answer the following multiple choice question. Respond with only the "
            "final line in the format 'ANSWER: $LETTER' (without quotes).\n\n"
            f"Question:\n{question}"
        )
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: $LETTER' (without "
        "quotes). Think step by step before answering.\n\n"
        f"Question:\n{question}"
    )


def extract_answer(text):
    m = _ANS_RE.search(text or "")
    if m:
        return m.group(1).upper(), True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        hits = _LAST_RE.findall(lines[-1])
        if hits:
            return hits[-1].upper(), True
    return "", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.direct:
        args.enable_thinking = False
    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit).copy()
    cases = []
    for _, r in df.iterrows():
        cases.append({
            "id": str(r["id"]),
            "category": str(r["category"]),
            "answer": str(r["answer"]).strip().upper(),
            "prompt": build_prompt(str(r["question"]), args.direct),
        })
    if args.dry_run:
        for c in cases[:2]:
            print("=" * 80)
            print(c["id"], c["category"], "gold=", c["answer"])
            print(c["prompt"][:1200])
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
    print(f"loaded {len(cases)}; done={len(done)}; todo={len(todo)}", flush=True)

    def one(c):
        t0 = time.time()
        rec = {"id": c["id"], "category": c["category"], "answer": c["answer"]}
        try:
            payload = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": c["prompt"]}],
            }
            if args.enable_thinking:
                payload["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["temperature"] = 0
            resp = post(args.api, payload)
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            pred, parsed = extract_answer(content)
            rec.update({
                "pred": pred, "parsed": parsed,
                "correct": int(parsed and pred == c["answer"]),
                "finish_reason": choice.get("finish_reason"),
                "ans": content.strip()[-600:],
                "usage": resp.get("usage", {}),
            })
        except Exception as e:
            rec.update({"pred": "", "parsed": False, "correct": 0, "error": repr(e)[:200]})
        rec["elapsed_s"] = round(time.time() - t0, 2)
        return rec

    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, c) for c in todo]), 1):
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
    trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
    errors = sum(1 for r in rows if "error" in r)
    toks = [r.get("usage", {}).get("completion_tokens", 0) for r in rows]
    secs = [r.get("elapsed_s", 0.0) for r in rows]
    by = {}
    for r in rows:
        b = by.setdefault(r["category"], [0, 0])
        b[0] += r["correct"]; b[1] += 1
    summary = {
        "n": n, "overall": correct / max(n, 1), "parsed_rate": parsed / max(n, 1),
        "truncated": trunc, "errors": errors,
        "mean_completion_tokens": sum(toks) / max(n, 1),
        "max_completion_tokens": max(toks) if toks else 0,
        "mean_elapsed_s": sum(secs) / max(n, 1),
        "max_elapsed_s": max(secs) if secs else 0,
        "by_category": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(by.items())},
        "protocol": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                     "reasoning_effort": args.reasoning_effort,
                     "enable_thinking": bool(args.enable_thinking),
                     "direct": bool(args.direct)},
    }
    out_sum = args.out.replace(".jsonl", "_summary.json")
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== HLE-MC: {correct}/{n} = {summary['overall']:.3f}; parsed={parsed}/{n}; "
          f"trunc={trunc}; errors={errors}; mean_tok={summary['mean_completion_tokens']:.0f}; "
          f"mean_s={summary['mean_elapsed_s']:.1f}; max_s={summary['max_elapsed_s']:.1f}")
    for k, v in summary["by_category"].items():
        print(f"  {k}: {v['correct']}/{v['n']}")
    print(f"wrote {out_sum}")


if __name__ == "__main__":
    main()
