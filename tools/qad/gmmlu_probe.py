#!/usr/bin/env python3
"""Global-MMLU-Lite probe (low-resource language tail): MCQ accuracy per language.

Protocol (fixed): chat endpoint, enable_thinking=false, temperature=0, max_tokens=16.
Prompt: question + options + "Answer with the single letter only."
Scoring: the first A/B/C/D letter in the response vs the gold answer.

Languages: the 1014 Tier1 low-resource tail (bn/sw/yo/my) + en reference.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(row: dict) -> tuple[str, str]:
    q = row.get("question") or ""
    opts = []
    for letter, key in [("A", "option_a"), ("B", "option_b"), ("C", "option_c"), ("D", "option_d")]:
        v = row.get(key)
        if v is not None:
            opts.append(f"{letter}. {v}")
    gold = str(row.get("answer", "")).strip()
    if gold in ("0", "1", "2", "3"):
        gold = "ABCD"[int(gold)]
    prompt = f"{q}\n\n" + "\n".join(opts) + "\n\nAnswer with the single letter only."
    return prompt, gold.upper()[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data")
    ap.add_argument("--langs", default="bn,sw,yo,my,en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    cases = []
    for lang in args.langs.split(","):
        t = pq.read_table(f"{args.data_dir}/gmmlu_{lang}.parquet")
        rows = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
        if args.limit:
            rows = rows[:args.limit]
        cases += [(lang, r) for r in rows]

    def one(item):
        lang, row = item
        prompt, gold = build_prompt(row)
        rec = {"lang": lang, "gold": gold, "subject": row.get("subject")}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 16,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": prompt}]})
            content = resp["choices"][0]["message"]["content"] or ""
            m = re.search(r"\b([ABCD])\b", content.upper())
            pred = m.group(1) if m else "?"
            rec.update({"pred": pred, "correct": int(pred == gold), "content": content[:80]})
        except Exception as e:
            rec.update({"error": repr(e)[:200], "correct": 0})
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    summary = {}
    n_done = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in cases]
        for fut in as_completed(futs):
            rec = fut.result()
            s = summary.setdefault(rec["lang"], {"n": 0, "correct": 0, "err": 0})
            s["n"] += 1
            s["correct"] += rec.get("correct", 0)
            s["err"] += int(bool(rec.get("error")))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f"  [{n_done}/{len(cases)}]", flush=True)

    print("=== summary (accuracy per language) ===")
    for lang, s in sorted(summary.items()):
        print(f"  {lang}: {s['correct']}/{s['n']} = {s['correct']/max(s['n'],1):.4f} (err={s['err']})")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
