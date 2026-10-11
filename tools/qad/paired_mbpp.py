#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mbpp paired 检验（1011）：从 --log_samples 落盘的 samples_mbpp jsonl 逐题 join。

依赖 samples 内 per-sample pass_at_1（lm_eval mbpp 评判链已带 EXTRACT_PATCH_1010B）。
join key = doc_hash（跨部署逐题对齐）；统计 = McNemar exact + paired bootstrap CI。

用法（容器内或任意 python3）：
  python3 paired_mbpp.py --a <dirA|file> --b <dirB|file> [--boot 10000] [--seed 0]
  dir 时自动挑目录下最新 samples_mbpp_*.jsonl
"""
import argparse
import glob
import json
import math
import os
import random


def pick(path):
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, "**", "samples_mbpp_*.jsonl"), recursive=True))
        if not fs:
            fs = sorted(glob.glob(os.path.join(path, "samples_mbpp_*.jsonl")))
        if not fs:
            raise SystemExit(f"samples_mbpp_*.jsonl not found under {path}")
        return fs[0], fs
    return path, [path]


def load(path, pickfs=None):
    recs = {}
    files = pickfs if pickfs else [path]
    n = 0
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            k = r.get("doc_hash") or r.get("prompt_hash") or r["doc"]["task_id"]
            recs[k] = (float(r["pass_at_1"]), n)
            n += 1
    return recs, n


def mcnemar_exact(b, c):
    # 双侧二项 p（b 与 c 不对称的精确检验）
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    m = b + c
    p = sum(math.comb(m, i) for i in range(0, k + 1)) * (2 ** -m) * 2
    return min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fa, files_a = pick(args.a)
    fb, files_b = pick(args.b)
    A, na = load(fa)
    B, nb = load(fb)
    keys = [k for k in A if k in B]
    pa = [A[k][0] for k in keys]
    pb = [B[k][0] for k in keys]
    n = len(keys)
    acc_a = sum(pa) / n
    acc_b = sum(pb) / n
    b = sum(1 for k in keys if A[k][0] == 1 and B[k][0] == 0)   # A 对 B 错
    c = sum(1 for k in keys if A[k][0] == 0 and B[k][0] == 1)   # B 对 A 错
    p = mcnemar_exact(b, c)
    rng = random.Random(args.seed)
    diffs = []
    for _ in range(args.boot):
        ia = [keys[rng.randrange(n)] for _ in range(n)]
        da = mean = sum(A[k][0] - B[k][0] for k in ia) / n
        diffs.append(da)
    diffs.sort()
    lo, hi = diffs[int(0.025 * args.boot)], diffs[min(args.boot - 1, int(0.975 * args.boot))]
    delta = acc_a - acc_b
    print(f"== paired mbpp: A={args.a}  B={args.b}")
    print(f"joined n={n} (a files={len(files_a)}, b files={len(files_b)})")
    print(f"acc A={acc_a:.4f}  acc B={acc_b:.4f}  (A-B)={delta*100:+.2f}pp")
    print(f"discordant: A=1B=0 -> {b}   A=0B=1 -> {c}   McNemar exact p={p:.4f}")
    print(f"paired bootstrap ({args.boot}, seed={args.seed}): 95%CI=[{lo*100:+.2f}pp, {hi*100:+.2f}pp]")
    verdict = "not significant" if (lo <= 0 <= hi) or p >= 0.05 else ("A better" if delta > 0 else "B better")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
