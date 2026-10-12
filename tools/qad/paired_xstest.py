#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XSTest paired 检验（1012）：samples csv（l0_xs / tgt_b1 / bf16_xs 等）逐题 join。

join key = (set, prompt)；refusal 列 = 判别器口径（关键词启发式）。
统计 = McNemar exact + paired bootstrap（refusal 差）。
用法：
  python3 paired_xstest.py --a <csvA> --b <csvB> [--set xstest_unsafe] [--boot 10000]
  --set 缺省 = xstest_unsafe（comply 差异判别主轴）；可 xstest_safe（over-refusal）
"""
import argparse
import csv
import math
import random


def load(path):
    recs = {}
    n_err = 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = (r["set"], r["prompt"])
        if r["refusal"] in (None, ""):
            n_err += 1
            continue
        recs[k] = int(r["refusal"])
    return recs, n_err


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (2.0 ** -n) * 2
    return min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--set", default="xstest_unsafe")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    def pick(path, st):
        recs, n_err = load(path)
        return {k[1]: v for k, v in recs.items() if k[0] == st}, n_err

    A, err_a = pick(args.a, args.set)
    B, err_b = pick(args.b, args.set)
    keys = [k for k in A if k in B]
    n = len(keys)
    acc_a = sum(A[k] for k in keys) / n
    acc_b = sum(B[k] for k in keys) / n
    b = sum(1 for k in keys if A[k] == 1 and B[k] == 0)
    c = sum(1 for k in keys if A[k] == 0 and B[k] == 1)
    p = mcnemar_exact(b, c)
    rng = random.Random(args.seed)
    diffs = []
    for _ in range(args.boot):
        ia = [keys[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(A[k] - B[k] for k in ia) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * args.boot)], diffs[min(args.boot - 1, int(0.975 * args.boot))]
    print(f"== paired xstest [{args.set}]: A={args.a}  B={args.b}")
    print(f"joined n={n} (errors a={err_a}, b={err_b})")
    print(f"refusal_rate A={acc_a:.4f}  B={acc_b:.4f}  (A-B)={(acc_a-acc_b)*100:+.2f}pp")
    print(f"discordant: A=1B=0 -> {b}   A=0B=1 -> {c}   McNemar exact p={p:.4f}")
    print(f"paired bootstrap ({args.boot}, seed={args.seed}): 95%CI=[{lo*100:+.2f}pp, {hi*100:+.2f}pp]")
    verdict = "not significant" if (lo <= 0 <= hi) or p >= 0.05 else ("A higher" if (acc_a - acc_b) > 0 else "B higher")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
