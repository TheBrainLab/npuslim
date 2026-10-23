#!/usr/bin/env python3
"""MMLU-Pro 500-question stratified probe (local knowledge axis).

Protocol (all arms): chat endpoint, enable_thinking=FALSE, temperature=0,
max_tokens=2048, 0-shot CoT prompt (AISBench-style), letter exact-match.

Input : mmlu_pro_500.parquet (question_id/question/options/answer/category)
Output: jsonl + _summary.json (overall, by_category, parse-fail, errors)

Usage:
  python3 mmlu_pro_probe.py --api http://HOST:PORT/v1/chat/completions \
      --model front_l0 --out .../mmlupro_l0.jsonl --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

CHOICES = "ABCDEFGHIJKLMNOP"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_ANS_RE = re.compile(r"(?i)ANSWER\s*:\s*([A-P])")
_LAST_LINE_RE = re.compile(r"(?i)\b([A-P])\b")


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_options(options):
    lines = []
    for i, opt in enumerate(options):
        s = str(opt).strip()
        if s == "N/A":
            continue
        lines.append(f"{CHOICES[i]}. {s}")
    return "\n".join(lines)


def build_prompt(question, options_str):
    return (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'ANSWER: $LETTER' (without quotes) where "
        "LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step "
        "before answering.\n\n"
        f"Question:\n{question}\n\nOptions:\n{options_str}"
    )


def extract_answer(text):
    m = _ANS_RE.search(text or "")
    if m:
        return m.group(1).upper(), True
    # fallback: last standalone letter on the last non-empty line
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        hits = _LAST_LINE_RE.findall(lines[-1])
        if hits:
            return hits[-1].upper(), True
    return "", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit).copy()
    cases = []
    for _, r in df.iterrows():
        opts = build_options(r["options"])
        cases.append({
            "question_id": int(r["question_id"]),
            "category": str(r["category"]),
            "answer": str(r["answer"]).strip().upper(),
            "prompt": build_prompt(str(r["question"]), opts),
        })

    if args.dry_run:
        for c in cases[:2]:
            print("=" * 80)
            print(c["question_id"], c["category"], "gold=", c["answer"])
            print(c["prompt"][:1200])
        print(f"dry-run: {len(cases)} cases; first prompt chars={len(cases[0]['prompt'])}")
        return

    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["question_id"]))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    todo = [c for c in cases if c["question_id"] not in done]
    print(f"loaded {len(cases)} cases; already done {len(done)}; todo {len(todo)}", flush=True)

    def one(c):
        t0 = time.time()
        rec = {"question_id": c["question_id"], "category": c["category"],
               "answer": c["answer"]}
        try:
            payload = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": c["prompt"]}],
            }
            if not args.enable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            resp = post(args.api, payload)
            content = (resp["choices"][0]["message"].get("content") or "")
            pred, parsed = extract_answer(content)
            rec.update({
                "pred": pred, "parsed": parsed, "correct": int(parsed and pred == c["answer"]),
                "ans": content.strip()[-500:],
                "usage": resp.get("usage", {}),
            })
        except Exception as e:
            rec.update({"pred": "", "parsed": False, "correct": 0, "error": repr(e)[:200]})
        rec["elapsed_s"] = round(time.time() - t0, 2)
        return rec

    res = []
    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for fut in as_completed(futs):
            r = fut.result()
            res.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if len(res) % 50 == 0:
                print(f"  [{len(res)}/{len(todo)}]", flush=True)

    all_rows = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass
    n = len(all_rows)
    correct = sum(r["correct"] for r in all_rows)
    parsed = sum(r["parsed"] for r in all_rows)
    errors = sum(1 for r in all_rows if "error" in r)
    by_cat = {}
    for r in all_rows:
        b = by_cat.setdefault(r["category"], [0, 0])
        b[0] += r["correct"]; b[1] += 1
    summary = {
        "n": n,
        "overall": correct / max(n, 1),
        "parsed_rate": parsed / max(n, 1),
        "errors": errors,
        "by_category": {k: {"correct": v[0], "n": v[1], "acc": v[0] / max(v[1], 1)}
                        for k, v in sorted(by_cat.items())},
        "protocol": {
            "enable_thinking": bool(args.enable_thinking),
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "prompt": "AISBench-style 0-shot CoT; letter exact-match",
        },
    }
    out_sum = args.out.replace(".jsonl", "_summary.json")
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== MMLU-Pro: {correct}/{n} = {summary['overall']:.3f}; "
          f"parsed={parsed}/{n}; errors={errors}")
    for k, v in summary["by_category"].items():
        print(f"  {k}: {v['correct']}/{v['n']} = {v['acc']:.3f}")
    print(f"wrote {out_sum}")


if __name__ == "__main__":
    main()
