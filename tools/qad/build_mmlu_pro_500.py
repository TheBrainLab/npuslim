#!/usr/bin/env python3
"""Build the 500-question stratified MMLU-Pro subset (fixed seed).

Proportional allocation by `category` (largest-remainder rounding), then a
fixed-seed sample per category. Writes a parquet + a JSON manifest for
reproducibility.

Usage:
  python3 build_mmlu_pro_500.py \
    --src .../mmlu_pro/mmlu_pro/test-00000-of-00001.parquet \
    --out .../mmlu_pro/mmlu_pro_500.parquet --n 500 --seed 1022
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

CHOICES = "ABCDEFGHIJKLMNOP"


def allocate(counts: pd.Series, n: int) -> dict:
    total = int(counts.sum())
    quotas = counts / total * n
    base = {k: int(math.floor(v)) for k, v in quotas.items()}
    rem = n - sum(base.values())
    # largest remainder; tie-break by category name for determinism
    order = sorted(quotas.index, key=lambda k: (-(quotas[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1022)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    needed = {"question_id", "question", "options", "answer", "category"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    if args.n > len(df):
        raise SystemExit(f"n={args.n} > rows={len(df)}")

    counts = df["category"].value_counts()
    alloc = allocate(counts, args.n)
    if sum(alloc.values()) != args.n:
        raise SystemExit(f"allocation sum {sum(alloc.values())} != {args.n}")

    parts = []
    for cat, k in sorted(alloc.items()):
        sub = df[df["category"] == cat]
        if k > len(sub):
            raise SystemExit(f"{cat}: alloc {k} > rows {len(sub)}")
        parts.append(sub.sample(n=k, random_state=args.seed))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if len(sample) != args.n:
        raise SystemExit(f"sample size {len(sample)} != {args.n}")
    if sample["question_id"].duplicated().any():
        raise SystemExit("duplicate question_id in sample")

    # Validate answer letters against the non-N/A option positions.
    bad = []
    for _, r in sample.iterrows():
        opts = [str(o) for o in r["options"]]
        valid = [CHOICES[i] for i, o in enumerate(opts) if o.strip() != "N/A"]
        if str(r["answer"]) not in valid:
            bad.append((r["question_id"], r["answer"], valid))
    if bad:
        raise SystemExit(f"invalid answer letters: {bad[:5]}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    sample.to_parquet(args.out, index=False)

    with open(args.src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": os.path.abspath(args.src),
        "source_sha256": src_sha,
        "source_rows": int(len(df)),
        "n": int(len(sample)),
        "seed": args.seed,
        "allocation": {k: int(v) for k, v in sorted(alloc.items())},
        "category_counts": {k: int(v) for k, v in sample["category"].value_counts().sort_index().items()},
        "question_ids": [int(x) for x in sample["question_id"].tolist()],
        "out": os.path.abspath(args.out),
    }
    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"wrote {args.out} ({len(sample)} rows)")
    print(f"wrote {manifest_path}")
    print("allocation:", json.dumps(manifest["allocation"], ensure_ascii=False))
    print("actual    :", json.dumps(manifest["category_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
