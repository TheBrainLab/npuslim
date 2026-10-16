#!/usr/bin/env python3
"""P1 router probe (built-in vLLM path).

Sends a fixed prompt to a vLLM server started with
``--enable-return-routed-experts`` and saves the per-token expert routing
returned in ``choices[0].routed_experts`` (base64 .npy, shape
``(num_tokens-1, num_layers, topk)``) to an .npz file.

Usage:
  python3 router_probe.py --api http://127.0.0.1:8000/v1/completions \
      --model glm52-w4a8-l0-router --tokenizer <path> --parquet <wikitext train parquet> \
      --tokens 2048 --out /llm_infer_workspace/.../router_dump/router_l0.npz
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    df = pd.read_parquet(args.parquet)
    col = "text" if "text" in df.columns else df.columns[0]
    text = "\n\n".join(str(x) for x in df[col].tolist())
    ids = tok.encode(text)[: args.tokens]
    print(f"[router-probe] prompt tokens = {len(ids)}", flush=True)

    payload = {"model": args.model, "prompt": ids, "max_tokens": 1,
               "temperature": 0, "echo": False}
    req = urllib.request.Request(args.api, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    b64 = ch.get("routed_experts")
    if not b64:
        raise SystemExit("[router-probe] routed_experts is None - was the server started "
                         "with --enable-return-routed-experts ?")
    arr = np.load(io.BytesIO(base64.b64decode(b64)))
    print(f"[router-probe] routed_experts shape = {arr.shape} dtype = {arr.dtype}", flush=True)
    np.savez_compressed(args.out, routed=arr, prompt_ids=np.asarray(ids, dtype=np.int64))
    print(f"[router-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
