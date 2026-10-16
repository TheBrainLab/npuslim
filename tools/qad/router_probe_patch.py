#!/usr/bin/env python3
"""P1 router probe (custom-hook path).

Sends a fixed 2048-token prompt to a vLLM server started with
``NPUSLIM_ROUTER_DUMP=...`` (the env-gated npuslim patch wraps
``DeepseekV2MoE.forward`` and dumps router top-k per layer). Requires
``--enforce-eager`` so the wrapper does not enter torch.compile/AOT.

The server writes the dump itself; this script only triggers one forward pass.
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", required=True)
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
               "temperature": 0, "echo": False}
    req = urllib.request.Request(args.api, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    print(f"[probe] OK id={d.get('id')}", flush=True)


if __name__ == "__main__":
    main()
