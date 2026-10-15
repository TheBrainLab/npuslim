#!/usr/bin/env python3
"""PPL–length curve + position-resolved NLL via a running vLLM completions API.

Implements the L0 instrument "PPL–length curve" (see
npuslim-docs/GLM-5.2-量化评测落地方案1015.md §2 L0 #6) and additionally the
per-position NLL profile that exposes length-dependent degradation
(quantization / RoPE rounding accumulate with context length).

Protocol
--------
* Build one long detokenized token stream from a wikitext parquet / pages.json.
* For each length L: cut the stream into non-overlapping windows of L tokens,
  request `echo=True, logprobs=1, max_tokens=1` with the window as *token ids*
  (exact, no re-tokenization), and sum the per-position logprob.
  Position 0 of each window carries no context -> excluded.
* PPL_L = exp(-sum(logp)/N).  Also records NLL by position bucket so a
  length-dependent drift inside the window is visible.

Runs inside a container that has `transformers` (and pandas/pyarrow for parquet).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def wikitext_detokenizer(string: str) -> str:
    import re
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    string = string.replace(" : ", ": ").replace(" ; ", "; ")
    string = string.replace(" . ", ". ").replace(" ! ", "! ")
    string = string.replace(" ? ", "? ").replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n").replace("\n ", "\n")
    string = string.replace(" N ", " 1 ").replace(" 's", "'s")
    return string


def build_stream(args, tok):
    if args.pages_json:
        pages = json.load(open(args.pages_json))
        texts = [str(p) for p in pages]
    else:
        import pandas as pd
        df = pd.read_parquet(args.parquet)
        col = "text" if "text" in df.columns else df.columns[0]
        texts = [str(x) for x in df[col].tolist()]
    texts = [wikitext_detokenizer(t) for t in texts]
    stream_text = "\n\n".join(texts)
    ids = tok.encode(stream_text)
    if args.limit_tokens and len(ids) > args.limit_tokens:
        ids = ids[: args.limit_tokens]
    return ids


def request_window(api, model, ids, timeout):
    payload = {
        "model": model,
        "prompt": ids,           # token ids -> exact, no re-tokenization
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "echo": True,
    }
    req = urllib.request.Request(
        api, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    lp = data["choices"][0]["logprobs"]["token_logprobs"]
    # layout: [null, lp(pos1), ..., lp(pos_{len-1}), gen_lp]
    return lp[1: len(ids)]     # exactly one logprob per prompt position >=1


def run_length(L, ids, api, model, concurrency, timeout, keep_profile, max_windows=0):
    nfull = len(ids) // L
    windows = [(i, ids[i * L:(i + 1) * L]) for i in range(nfull)]
    if max_windows and len(windows) > max_windows:
        windows = windows[:max_windows]
    nll_sum = 0.0
    n_tok = 0
    pos_sum = None
    pos_cnt = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(request_window, api, model, w, timeout): i for i, w in windows}
        done = 0
        for f in futs:
            lps = f.result()
            for j, v in enumerate(lps):          # widget position j (0-based) -> token j+1
                if v is None:
                    continue
                nll_sum += -v
                n_tok += 1
                if keep_profile:
                    if pos_sum is None:
                        pos_sum = [0.0] * len(lps)
                        pos_cnt = [0] * len(lps)
                    pos_sum[j] += -v
                    pos_cnt[j] += 1
            done += 1
            if done % 50 == 0:
                print(f"    L={L} windows {done}/{nfull} elapsed={time.time()-t0:.0f}s", flush=True)
    ppl = math.exp(nll_sum / n_tok) if n_tok else float("nan")
    prof = None
    if keep_profile and pos_sum:
        prof = [{"pos_bin": j, "nll": pos_sum[j] / pos_cnt[j], "n": pos_cnt[j]}
                for j in range(len(pos_sum)) if pos_cnt[j]]
    return {"L": L, "n_windows": nfull, "n_tokens": n_tok,
            "mean_nll": nll_sum / n_tok, "ppl": ppl, "elapsed_s": time.time() - t0,
            "profile": prof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", default="glm52-w4a8-eval")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--pages-json", default=None)
    ap.add_argument("--lengths", default="4096,16384,65536,131072,262140")
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="record position-resolved NLL")
    ap.add_argument("--profile-max-len", type=int, default=131072)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"[ppl] tokenizer loaded ({args.tokenizer})", flush=True)

    ids = build_stream(args, tok)
    print(f"[ppl] stream tokens = {len(ids)}", flush=True)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = []
    for L in lengths:
        if L > len(ids):
            print(f"[ppl] L={L} > stream ({len(ids)}) - skip", flush=True)
            continue
        keep = args.profile and L <= args.profile_max_len
        print(f"[ppl] L={L} (windows={len(ids)//L}, profile={keep})", flush=True)
        r = run_length(L, ids, args.api, args.model, args.concurrency, args.timeout, keep, args.max_windows)
        print(f"[ppl]   L={r['L']:>7} ppl={r['ppl']:.4f} nll={r['mean_nll']:.4f} "
              f"tok={r['n_tokens']} {r['elapsed_s']:.0f}s", flush=True)
        results.append(r)

    payload = {"model": args.model, "stream_tokens": len(ids),
               "tokenizer": args.tokenizer, "results": results}
    if args.out:
        json.dump(payload, open(args.out, "w"), indent=2)
        print(f"[ppl] wrote {args.out}", flush=True)
    print("\n================= PPL–LENGTH =================")
    for r in results:
        print(f"  L={r['L']:>7}  ppl={r['ppl']:.4f}  mean_nll={r['mean_nll']:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()
