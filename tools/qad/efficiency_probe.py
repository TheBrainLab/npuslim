#!/usr/bin/env python3
"""Efficiency probe: decode throughput (conc x 512 tokens) + prefill latency (~4.5K prompt -> 1 token).

Protocol: chat endpoint, enable_thinking=false, temperature=0. Same workload for all arms.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PROMPT = ("Write a detailed explanation of how a bicycle works, covering the frame, "
          "wheels, gears, and brakes. ") * 3


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=512)
    args = ap.parse_args()
    out = {}

    base = {"model": args.model, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    # warmup
    post(args.api, {**base, "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})

    # (a) decode throughput
    payload = {**base, "max_tokens": args.gen_tokens,
               "messages": [{"role": "user", "content": PROMPT}]}

    def one(_):
        t0 = time.time()
        r = post(args.api, payload)
        dt = time.time() - t0
        return dt, r.get("usage", {}).get("completion_tokens", 0)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        res = list(ex.map(one, range(args.conc)))
    wall = time.time() - t0
    toks = sum(n for _, n in res)
    out["decode"] = {"conc": args.conc, "wall_s": round(wall, 2), "tokens": toks,
                     "tok_per_s": round(toks / wall, 1),
                     "per_req_s": [round(d, 2) for d, _ in res]}

    # (b) prefill latency: long prompt -> 1 token
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 400
    payload2 = {**base, "max_tokens": 1, "messages": [{"role": "user", "content": long_prompt}]}
    t0 = time.time()
    post(args.api, payload2)
    out["prefill"] = {"prompt_chars": len(long_prompt), "wall_s": round(time.time() - t0, 2)}

    print(json.dumps(out, indent=1))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
