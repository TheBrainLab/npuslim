#!/usr/bin/env python3
"""Build text-only HLE exact-match subsets (20 smoke / 120 full candidate).

Filters:
  * answer_type == "exactMatch"
  * image is empty (text-only)
  * answer length <= 50 (keeps short, automatable answers)

Allocation: min_per_category + proportional remainder (largest remainder),
fixed seed. Writes parquet + JSON manifests.

Usage:
  python3 build_hle_subsets.py --src .../hle/test.parquet \
      --out-dir .../hle --seed 1022
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd


def allocate(counts: pd.Series, n: int, min_per_category: int) -> dict:
    cats = list(counts.index)
    alloc = {c: min_per_category for c in cats}
    remaining = n - sum(alloc.values())
    if remaining < 0:
        raise SystemExit(f"n={n} too small for {len(cats)} categories × min={min_per_category}")
    pool = {c: max(int(counts[c]) - min_per_category, 0) for c in cats}
    total = sum(pool.values())
    if total == 0:
        return alloc
    quotas = {c: pool[c] / total * remaining for c in cats}
    base = {c: int(math.floor(quotas[c])) for c in cats}
    rem = remaining - sum(base.values())
    order = sorted(cats, key=lambda c: (-(quotas[c] - base[c]), c))
    for c in order[:rem]:
        base[c] += 1
    for c in cats:
        alloc[c] += base[c]
    return alloc


def build(df, n, min_per_category, seed, out_path, max_answer_len):
    counts = df["category"].value_counts()
    alloc = allocate(counts, n, min_per_category)
    if sum(alloc.values()) != n:
        raise SystemExit(f"allocation sum {sum(alloc.values())} != {n}")
    parts = []
    for cat, k in sorted(alloc.items()):
        sub = df[df["category"] == cat]
        if k > len(sub):
            raise SystemExit(f"{cat}: alloc {k} > rows {len(sub)}")
        parts.append(sub.sample(n=k, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(sample) != n or sample["id"].duplicated().any():
        raise SystemExit(f"bad sample: n={len(sample)} dup={sample['id'].duplicated().any()}")
    sample.to_parquet(out_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": int(len(sample)),
        "seed": seed,
        "min_per_category": min_per_category,
        "max_answer_len": max_answer_len,
        "allocation": {k: int(v) for k, v in sorted(alloc.items())},
        "category_counts": {k: int(v) for k, v in sample["category"].value_counts().sort_index().items()},
        "ids": sample["id"].tolist(),
        "out": os.path.abspath(out_path),
    }
    with open(os.path.splitext(out_path)[0] + "_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path} ({len(sample)}) alloc={manifest['allocation']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1022)
    ap.add_argument("--max-answer-len", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    text = df[
        (df["answer_type"] == "exactMatch")
        & (df["image"].astype(str).str.len() == 0)
        & (df["answer"].astype(str).str.len() <= args.max_answer_len)
    ].copy()
    text = text[text["answer"].astype(str).str.strip() != ""]
    print(f"text-only exactMatch (<={args.max_answer_len} chars): {len(text)}")
    with open(args.src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"source sha256: {src_sha}")
    os.makedirs(args.out_dir, exist_ok=True)
    build(text, 20, 1, args.seed, os.path.join(args.out_dir, "hle_smoke20.parquet"), args.max_answer_len)
    build(text, 120, 3, args.seed, os.path.join(args.out_dir, "hle_120.parquet"), args.max_answer_len)


if __name__ == "__main__":
    main()
