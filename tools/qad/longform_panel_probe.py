#!/usr/bin/env python3
"""Long-form generation damage panel (1014 Tier1).

30 fixed prompts asking for long structured answers (>=800 words, headings,
lists). Protocol: chat endpoint, enable_thinking=false, temperature=0, max_tokens=2048.

Metrics (per product):
  * truncation rate : finish_reason == "length" (hit the cap)
  * mean chars / mean words
  * distinct-2      : distinct bigrams / total bigrams
  * rep-4gram       : 1 - distinct 4-grams / total 4-grams (repetition)
  * no-claim rate   : hedging/refusal family in the response
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOPICS = [
    "how to grow tomatoes in a small balcony garden",
    "the history of the internet from ARPANET to today",
    "a practical guide to personal budgeting for students",
    "how electric vehicles work, from battery to motor",
    "the causes and consequences of the 2008 financial crisis",
    "how to prepare for a marathon in six months",
    "the water cycle and its role in climate",
    "a beginner's guide to machine learning concepts",
    "the history of the Silk Road and its trade goods",
    "how modern cryptography protects online banking",
    "the biology of sleep and why it matters",
    "how to cook a week of healthy meals in one afternoon",
    "the physics of sound and how musical instruments work",
    "a guide to visiting national parks responsibly",
    "the history of the printing press and its impact",
    "how vaccines train the immune system",
    "the economics of supply chains and shipping",
    "how to learn a foreign language as an adult",
    "the geology of volcanoes and earthquake zones",
    "a practical introduction to home composting",
    "the history of space exploration milestones",
    "how air traffic control keeps flights safe",
    "the basics of photography composition",
    "how antibiotics work and why resistance matters",
    "the history and culture of tea",
    "how to set up a small solar power system",
    "the science of weather forecasting",
    "a guide to reading financial statements",
    "the history of the Olympic Games",
    "how noise-cancelling headphones work",
]

NO_CLAIM = [re.compile(p, re.I) for p in [
    r"\bI can'?t (write|provide|help with)\b", r"\bI cannot (write|provide|help with)\b",
    r"\bI'?m unable to\b", r"\bI am unable to\b", r"\bI won'?t\b",
    r"抱歉[^。]{0,30}(不能|无法)", r"我不能", r"我无法",
]]


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    prompts = [f"Write a comprehensive guide to {t}. Requirements: at least 800 words, "
               f"organized into sections with markdown headings (##), and include at least "
               f"two bullet lists." for t in TOPICS]

    def one(prompt):
        rec = {"topic": prompt[26:80]}
        try:
            resp = post(args.api, {"model": args.model, "temperature": 0, "max_tokens": 2048,
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "messages": [{"role": "user", "content": prompt}]})
            ch = resp["choices"][0]
            content = ch["message"]["content"] or ""
            toks = re.findall(r"\w+", content.lower())
            d2 = len(set(ngrams(toks, 2))) / max(len(ngrams(toks, 2)), 1)
            g4 = ngrams(toks, 4)
            rep4 = 1 - len(set(g4)) / max(len(g4), 1)
            rec.update({"finish": ch.get("finish_reason"),
                        "chars": len(content), "words": len(toks),
                        "distinct2": round(d2, 4), "rep4": round(rep4, 4),
                        "no_claim": int(any(p.search(content[:600]) for p in NO_CLAIM)),
                        "headings": content.count("\n##"), "bullets": content.count("\n- ") + content.count("\n* "),
                        "sample": content[:150]})
        except Exception as e:
            rec.update({"error": repr(e)[:200]})
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    agg = {"n": 0, "trunc": 0, "chars": 0, "words": 0, "d2": 0.0, "rep4": 0.0, "nc": 0, "err": 0,
           "headings": 0, "bullets": 0}
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, p) for p in prompts]
        for fut in as_completed(futs):
            r = fut.result()
            agg["n"] += 1
            if r.get("error"):
                agg["err"] += 1
            else:
                agg["trunc"] += int(r.get("finish") == "length")
                agg["chars"] += r.get("chars", 0)
                agg["words"] += r.get("words", 0)
                agg["d2"] += r.get("distinct2", 0)
                agg["rep4"] += r.get("rep4", 0)
                agg["nc"] += r.get("no_claim", 0)
                agg["headings"] += int(r.get("headings", 0) > 0)
                agg["bullets"] += int(r.get("bullets", 0) >= 2)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
    n = max(agg["n"] - agg["err"], 1)
    print("=== long-form panel summary ===")
    print(f"  n={agg['n']} err={agg['err']}")
    print(f"  truncation = {agg['trunc']}/{n} = {agg['trunc']/n:.3f}")
    print(f"  mean chars = {agg['chars']/n:.0f} | mean words = {agg['words']/n:.0f}")
    print(f"  distinct-2 = {agg['d2']/n:.4f} | rep-4gram = {agg['rep4']/n:.4f}")
    print(f"  no-claim = {agg['nc']}/{n} = {agg['nc']/n:.3f}")
    print(f"  headings ok = {agg['headings']}/{n} | >=2 bullet lists = {agg['bullets']}/{n}")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(agg, f, indent=1)


if __name__ == "__main__":
    main()
