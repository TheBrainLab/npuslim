#!/usr/bin/env python3
"""P1 router-dump probe: send a fixed 2048-token prompt to trigger one forward.

Used with the env-gated `router_dump` npuslim patch (NPUSLIM_ROUTER_DUMP=...):
one prefill call per MoE layer records its router top-k, so a single request
produces a complete routing snapshot for the prompt.

Usage:
  python3 router_dump_probe.py --api http://127.0.0.1:8000/v1/completions \
      --model glm52-w4a8-eval --parquet <wikitext train parquet> --tokens 2048
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", default="glm52-w4a8-eval")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    df = pd.read_parquet(args.parquet)
    col = "text" if "text" in df.columns else df.columns[0]
    text = "\n\n".join(str(x) for x in df[col].tolist())
    ids = tok.encode(text)[: args.tokens]
    print(f"[probe] prompt tokens = {len(ids)}", flush=True)

    payload = {"model": args.model, "prompt": ids, "max_tokens": 1,
               "temperature": 0, "logprobs": 1, "echo": False}
    req = urllib.request.Request(args.api, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    print(f"[probe] OK id={d.get('id')} usage={d.get('usage')}", flush=True)


if __name__ == "__main__":
    main()
