#!/usr/bin/env python3
"""Aggregate L0-fidelity + PPL-length JSONs into a markdown table.

Usage: python3 l0_report_1015.py --dir <results_dir> [--tags l0,p2clip]
"""
import argparse
import json
import os


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tags", default="l0")
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]

    print("### Distribution fidelity vs BF16 teacher cache\n")
    print("| product | n_pos | NLL gap | PPL ratio | top-1 agr | top-5 ovl | KL(top-20) | margin c | margin b |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in tags:
        d = load(os.path.join(args.dir, f"dist_fidelity_{t}.json"))
        if not d:
            print(f"| {t} | - | - | - | - | - | - | - | - |"); continue
        print(f"| {t} | {d['n_positions']} | {d['nll_gap_mean']:.4f} | {d['ppl_ratio']:.4f} | "
              f"{d['top1_agreement']:.4f} | {d['top5_overlap_mean']:.3f} | {d['kl_topk_proxy_mean']:.4f} | "
              f"{d['margin_coef_c']} | {d['margin_intercept_b']} |")

    print("\n### PPL–length curve\n")
    print("| product | corpus | " + " | ".join(["L"]))
    # print per-corpus rows
    for t in tags:
        for corpus in ("val", "train"):
            d = load(os.path.join(args.dir, f"ppl_length_{corpus}_{t}.json"))
            if not d:
                continue
            cells = []
            for r in d["results"]:
                cells.append(f"{r['L']}: {r['ppl']:.4f}")
            print(f"| {t} | {corpus} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
