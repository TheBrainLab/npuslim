#!/usr/bin/env python3
"""Build HLE multiple-choice subsets (10 smoke / 120 candidate).

Filter: answer_type == "multipleChoice" and image empty (text-only).
Question text already contains "Answer Choices: ..."; answer is a letter.
Allocation: min_per_category + proportional remainder, fixed seed.

Usage:
  python3 build_hle_mc_subsets.py --src .../hle/test.parquet \
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
    pool = {c: max(int(counts[c]) - min_per_category, 0) for c in cats}
    total = sum(pool.values())
    if remaining < 0 or total == 0:
        raise SystemExit(f"bad allocation n={n} min={min_per_category}")
    quotas = {c: pool[c] / total * remaining for c in cats}
    base = {c: int(math.floor(quotas[c])) for c in cats}
    rem = remaining - sum(base.values())
    for c in sorted(cats, key=lambda x: (-(quotas[x] - base[x]), x))[:rem]:
        base[c] += 1
    for c in cats:
        alloc[c] += base[c]
    return alloc


def build(df, n, min_per_category, seed, out_path):
    alloc = allocate(df["category"].value_counts(), n, min_per_category)
    if sum(alloc.values()) != n:
        raise SystemExit(f"allocation sum {sum(alloc.values())} != {n}")
    parts = []
    for cat, k in sorted(alloc.items()):
        sub = df[df["category"] == cat]
        if k > len(sub):
            raise SystemExit(f"{cat}: alloc {k} > rows {len(sub)}")
        parts.append(sub.sample(n=k, random_state=seed))
    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(sample) != n or sample["id"].duplicated().any():
        raise SystemExit("bad sample")
    sample.to_parquet(out_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": int(n),
        "seed": seed,
        "min_per_category": min_per_category,
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
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    mc = df[
        (df["answer_type"] == "multipleChoice")
        & (df["image"].astype(str).str.len() == 0)
    ].copy()
    print(f"text-only multipleChoice: {len(mc)}")
    with open(args.src, "rb") as f:
        print("source sha256:", hashlib.sha256(f.read()).hexdigest())
    os.makedirs(args.out_dir, exist_ok=True)
    build(mc, 10, 1, args.seed, os.path.join(args.out_dir, "hle_mc_smoke10.parquet"))
    build(mc, 60, 3, args.seed, os.path.join(args.out_dir, "hle_mc_60.parquet"))
    build(mc, 200, 3, args.seed, os.path.join(args.out_dir, "hle_mc_200.parquet"))
    build(mc, 120, 3, args.seed, os.path.join(args.out_dir, "hle_mc_120.parquet"))


if __name__ == "__main__":
    main()
