#!/usr/bin/env python3
"""P1 router comparison (built-in vLLM routed-experts format).

Loads two .npz files produced by `router_probe.py` (arrays of shape
``(tokens, layers, topk)``) and reports, per layer:
  * top-1 agreement   (fraction of tokens with the same 1st-choice expert)
  * top-k set overlap (mean |A∩B|/k)
  * expert-usage KL   (histogram over experts, Laplace-smoothed)
  * usage entropy

Usage:
  python3 router_compare.py --base router_l0.npz --target router_p2clip.npz [--target ...]
"""
from __future__ import annotations

import argparse

import numpy as np


def load(path):
    """Load either the built-in .npz (tokens,layers,topk) or the patch .pt dict."""
    if path.endswith(".npz"):
        z = np.load(path)
        return z["routed"], (z["prompt_ids"] if "prompt_ids" in z else None)
    import torch

    d = torch.load(path, map_location="cpu")
    layers = d["layers"]
    idxs = sorted(layers)
    arrs = []
    for i in idxs:
        calls = layers[i]
        # pick the call with the MOST tokens (the real request, not the 1-token warmup)
        best = max(calls, key=lambda t: t.shape[0])
        arrs.append(best.numpy() if hasattr(best, "numpy") else np.asarray(best))
    P = min(a.shape[0] for a in arrs)
    k = min(a.shape[1] for a in arrs)
    arr = np.stack([a[:P, :k] for a in arrs], axis=1)  # (P, L, k)
    return arr, None


def usage_hist(ids, n_experts=256):
    h = np.bincount(ids.reshape(-1).astype(np.int64), minlength=n_experts).astype(np.float64)
    return h / max(1.0, h.sum())


def kl(p, q, eps=1e-9):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def entropy(p, eps=1e-12):
    p = p + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def compare(base, target, tag):
    T = min(base.shape[0], target.shape[0])
    L = min(base.shape[1], target.shape[1])
    k = min(base.shape[2], target.shape[2])
    a, b = base[:T, :L, :k], target[:T, :L, :k]
    ag1, ov, kls, eb, et = [], [], [], [], []
    for li in range(L):
        x, y = a[:, li, :], b[:, li, :]
        ag1.append(float((x[:, 0] == y[:, 0]).mean()))
        inter = np.array([len(set(x[i].tolist()) & set(y[i].tolist())) for i in range(T)],
                         dtype=np.float64)
        ov.append(float((inter / k).mean()))
        hx, hy = usage_hist(x), usage_hist(y)
        kls.append(kl(hx, hy))
        eb.append(entropy(hx)); et.append(entropy(hy))
    r = {
        "tag": tag, "tokens": T, "layers": L, "topk": k,
        "top1_agreement_mean": float(np.mean(ag1)),
        "top1_agreement_min": float(np.min(ag1)),
        "topk_overlap_mean": float(np.mean(ov)),
        "usage_kl_mean": float(np.mean(kls)),
        "usage_kl_max": float(np.max(kls)),
        "entropy_base": float(np.mean(eb)),
        "entropy_target": float(np.mean(et)),
    }
    print(f"\n===== {tag} =====")
    for kk, vv in r.items():
        print(f"  {kk:22}: {vv}")
    # worst layers by top-1 agreement
    order = np.argsort(ag1)
    print("  worst 5 layers (top-1 agr):", [(int(i), round(ag1[i], 3)) for i in order[:5]])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b, _ = load(args.base)
    print(f"base: shape={b.shape} dtype={b.dtype}")
    results = []
    for t in args.target:
        d, _ = load(t)
        r = compare(b, d, t.split("/")[-1])
        if r:
            results.append(r)
    if args.out:
        import json
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
