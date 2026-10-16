#!/usr/bin/env python3
"""P2a builder: mbpp test rows + a fixed long 'prefix' context (for long-context code eval).

Produces a JSON usable by the `mbpp_longctx` lm_eval task (see
glm52-qad/mbpp_longctx/): each row carries the original mbpp fields plus a
`prefix` string of exactly `--tokens` tokens (drawn from the code corpus), which
the task template prepends to the mbpp prompt. This isolates the effect of
context length on code ability (same 500 problems, same tests, same extraction).

Run inside the eval container (needs `datasets` + `transformers` + HF cache).
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=32768)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    print(f"[build] mbpp test rows = {len(ds)}", flush=True)

    chunks = []
    with open(args.corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages") or row.get("conversations") or []
            for m in msgs:
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    chunks.append(c)
            if len(tok.encode("\n\n".join(chunks))) > args.tokens + 4000:
                break
    ids = tok.encode("\n\n".join(chunks))[: args.tokens]
    prefix = tok.decode(ids)
    print(f"[build] prefix tokens = {len(ids)} chars = {len(prefix)}", flush=True)

    rows = []
    for r in ds:
        rows.append({
            "task_id": r["task_id"],
            "text": r["text"],
            "test_list": list(r["test_list"]),
            "code": r["code"],
            "prefix": prefix,
        })
    json.dump(rows, open(args.out, "w"))
    print(f"[build] wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
