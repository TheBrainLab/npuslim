#!/usr/bin/env python3
"""#5 P2d: long-context robustness probe — local NIAH (RULER-style), 64K / 128K.

Design (fixed protocol; relative comparison across products is what matters):
  * haystack  = wikitext-2 train text, ~4 chars/token approximation for the target length
  * needle(s) = "One of the special magic numbers for {key} is: {value}." inserted at
                controlled depths (single-key: 1 needle; multi-key: 4 needles)
  * query     = "What is the special magic number for {key} mentioned in the provided text?"
  * metric    = retrieval accuracy (the value string appears in the answer)

Runs on the control host (urllib only); talks to the vLLM server over the node IP.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request

WIKITEXT = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/wikitext_train.txt"
CHARS_PER_TOKEN = 4.0
DEPTHS_SINGLE = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
KEYS_MULTI = ["alpha", "beta", "gamma", "delta"]
DEPTHS_MULTI = [0.1, 0.35, 0.6, 0.9]


def load_haystack(path: str) -> str:
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        col = "text" if "text" in df.columns else df.columns[0]
        return "\n".join(str(x) for x in df[col].tolist())
    except Exception:
        # fallback: any large local text file
        for cand in ("/home/yult/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx/mbpp_longctx_32k.json",):
            try:
                d = json.load(open(cand))
                return "\n".join(r.get("prefix", "") for r in d[:40])
            except Exception:
                pass
        raise SystemExit("no haystack source available")


# Local API calls must bypass any host proxy (坑51: 10.1.0.x 会被死代理劫持 -> 502)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_case(hay: str, n_chars: int, needles: list[tuple[str, str]], depths: list[float], query: str) -> str:
    """Insert needles at the given depth fractions of an n_chars haystack."""
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
    return f"{text}\n\nQuestion: {query}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="65536,131072")
    ap.add_argument("--per-length", type=int, default=7, help="single-key cases per length")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hay = load_haystack(WIKITEXT)
    print(f"[niah] haystack chars = {len(hay)}", flush=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    results = []
    for L in lengths:
        n_chars = int(L * CHARS_PER_TOKEN)
        if n_chars > len(hay):
            n_chars = len(hay)
        # single-key cases
        for i in range(args.per_length):
            depth = DEPTHS_SINGLE[i % len(DEPTHS_SINGLE)]
            key = f"{rng.randint(1000, 9999)}"
            val = str(rng.randint(100000, 999999))
            q = f"What is the special magic number for {key} mentioned in the provided text?"
            prompt = build_case(hay, n_chars, [(key, val)], [depth], q)
            results.append({"length": L, "kind": "single", "key": key, "value": val, "depth": depth, "prompt": prompt})
        # multi-key cases
        for i in range(max(2, args.per_length // 3)):
            vals = {k: str(rng.randint(100000, 999999)) for k in KEYS_MULTI}
            ask = rng.choice(KEYS_MULTI)
            q = f"What is the special magic number for {ask} mentioned in the provided text?"
            prompt = build_case(hay, n_chars, list(vals.items()), DEPTHS_MULTI, q)
            results.append({"length": L, "kind": "multi", "key": ask, "value": vals[ask], "depth": -1, "prompt": prompt})

    summary = {}

    def one(rec):
        payload = {"model": args.model, "temperature": 0, "max_tokens": 64,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": rec["prompt"]}]}
        rec_out = {k: v for k, v in rec.items() if k != "prompt"}
        try:
            resp = post(args.api, payload)
            ans = resp["choices"][0]["message"]["content"]
            hit = rec["value"] in ans
            rec_out.update({"hit": int(hit), "answer": ans.strip()[:200], "prompt_chars": len(rec["prompt"])})
        except Exception as e:
            rec_out.update({"hit": 0, "error": repr(e)[:200]})
        return rec_out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_done = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, r) for r in results]
        for fut in as_completed(futs):
            rec_out = fut.result()
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            fout.flush()
            k = f"L{rec_out['length']}_{rec_out['kind']}"
            s = summary.setdefault(k, {"n": 0, "hit": 0})
            s["n"] += 1
            s["hit"] += rec_out["hit"]
            n_done += 1
            print(f"  [{n_done}/{len(results)}] {k} key={rec_out['key']} hit={rec_out['hit']}", flush=True)

    print("=== summary ===")
    for k, s in sorted(summary.items()):
        print(f"  {k}: {s['hit']}/{s['n']} = {s['hit']/max(s['n'],1):.3f}")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
