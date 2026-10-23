#!/usr/bin/env python3
"""Re-score the 1019 OneRuler runs without re-inference.

Fixes the `niah_none` scoring bug: the original probe hard-coded the English
string "none", while OneRuler's gold answers are localized. Also reports CWE
mean recall (the metric the probe docstring intended) alongside all-or-nothing
`correct`.

Inputs : glm52-qad/audit1009/frontier_{arm}_oneruler.jsonl
Outputs: frontier_{arm}_oneruler_rescored.jsonl
         frontier_{arm}_oneruler_summary_fixed.json
         oneruler_fix_summary_1022.json / .md
"""
from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd

DATA_DIR = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/frontier/oneruler"
AUDIT_DIR = "/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009"
ARMS = ["bf16", "l0", "w4a8c8", "w8a8", "p2clip"]
LANGS = ["en", "zh", "hi", "sw", "st", "ta"]


def load_niah_none_gold(data_dir):
    """niah_none gold is constant per language across lengths; assert it."""
    gold = {}
    for lang in LANGS:
        vals = set()
        for length in (4096, 8192, 16384):
            p = os.path.join(data_dir, f"{lang}_{length}.parquet")
            df = pd.read_parquet(p)
            sub = df[df["task"] == "niah_none"]
            for a in sub["answer"]:
                vals.add(tuple(str(x) for x in a))
        if len(vals) != 1:
            raise SystemExit(f"{lang}: niah_none gold not constant: {vals}")
        gold[lang] = list(next(iter(vals)))
    return gold


def summarize(rows):
    summary = {
        "n": len(rows),
        "by_task": {},
        "by_lang": {},
        "by_length": {},
        "by_task_lang": {},
        "by_task_recall": {},
    }
    for r in rows:
        for key, bucket in (("task", "by_task"), ("lang", "by_lang"), ("length", "by_length")):
            b = summary[bucket].setdefault(r[key], [0, 0])
            b[0] += r["correct"]
            b[1] += 1
        b = summary["by_task_lang"].setdefault(f"{r['task']}|{r['lang']}", [0, 0])
        b[0] += r["correct"]
        b[1] += 1
        summary["by_task_recall"].setdefault(r["task"], []).append(float(r.get("recall", 0.0)))
    summary["by_task_recall"] = {
        k: sum(v) / len(v) for k, v in summary["by_task_recall"].items()
    }
    summary["overall"] = sum(r["correct"] for r in rows) / max(len(rows), 1)
    summary["overall_recall"] = sum(r.get("recall", 0.0) for r in rows) / max(len(rows), 1)
    return summary


def z_two_prop(a, n1, b, n2):
    p = (a + b) / max(n1 + n2, 1)
    se = math.sqrt(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    return (a / max(n1, 1) - b / max(n2, 1)) / se if se else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--audit-dir", default=AUDIT_DIR)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    gold = load_niah_none_gold(args.data_dir)
    combined = {
        "generated_by": "oneruler_rescore_1019.py",
        "fix": "niah_none scored against localized gold answers instead of hard-coded 'none'",
        "arms": {},
    }

    for arm in args.arms.split(","):
        src = os.path.join(args.audit_dir, f"frontier_{arm}_oneruler.jsonl")
        if not os.path.exists(src):
            print(f"WARN missing {src}")
            continue
        rows = [json.loads(line) for line in open(src, encoding="utf-8")]
        fixed = []
        for r in rows:
            r = dict(r)
            r["correct_old"] = r["correct"]
            if r["task"] == "niah_none":
                hit = int(any(g.lower() in r["ans"].lower() for g in gold[r["lang"]]))
                r["hits"], r["total"], r["correct"] = hit, 1, hit
                r["answers"] = gold[r["lang"]]
            r["recall"] = (r["hits"] / r["total"]) if r["total"] else 0.0
            fixed.append(r)

        out_jsonl = os.path.join(args.audit_dir, f"frontier_{arm}_oneruler_rescored.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in fixed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = summarize(fixed)
        out_sum = os.path.join(args.audit_dir, f"frontier_{arm}_oneruler_summary_fixed.json")
        with open(out_sum, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)

        old_correct = sum(r["correct_old"] for r in fixed)
        old_overall = old_correct / len(fixed)
        niah_none = summary["by_task"].get("niah_none", [0, 0])
        combined["arms"][arm] = {
            "n": len(fixed),
            "old_overall": old_overall,
            "fixed_overall": summary["overall"],
            "delta_pp": 100 * (summary["overall"] - old_overall),
            "niah_none_fixed": f"{niah_none[0]}/{niah_none[1]}",
            "cwe_recall": summary["by_task_recall"].get("cwe"),
            "cwe_easy_recall": summary["by_task_recall"].get("cwe_easy"),
            "by_task": summary["by_task"],
            "by_task_recall": summary["by_task_recall"],
            "summary_json": out_sum,
            "rescored_jsonl": out_jsonl,
        }
        print(f"{arm:8s} old={old_overall:.4f} fixed={summary['overall']:.4f} "
              f"({100*(summary['overall']-old_overall):+.1f}pp)  niah_none={niah_none[0]}/{niah_none[1]}")

    # l0 vs pooled other four arms, fixed overall
    if "l0" in combined["arms"]:
        l0 = combined["arms"]["l0"]
        others = [a for a in combined["arms"] if a != "l0"]
        oa = sum(combined["arms"][a]["fixed_overall"] * combined["arms"][a]["n"] for a in others)
        on = sum(combined["arms"][a]["n"] for a in others)
        diff = l0["fixed_overall"] - oa / on
        z = z_two_prop(l0["fixed_overall"] * l0["n"], l0["n"], oa, on)
        combined["l0_vs_others"] = {
            "diff_pp": 100 * diff,
            "z": z,
            "significant_0.05": abs(z) >= 1.96,
        }
        print(f"l0 vs others: {100*diff:+.1f}pp  z={z:+.2f}")

    out = os.path.join(args.audit_dir, "oneruler_fix_summary_1022.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=1)

    md = os.path.join(args.audit_dir, "oneruler_fix_summary_1022.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("| arm | old overall | fixed overall | delta | niah_none fixed | CWE recall | CWE-easy recall |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for arm in ["bf16", "l0", "w4a8c8", "w8a8", "p2clip"]:
            d = combined["arms"].get(arm)
            if not d:
                continue
            f.write(f"| {arm} | {100*d['old_overall']:.1f}% | {100*d['fixed_overall']:.1f}% | "
                    f"{d['delta_pp']:+.1f}pp | {d['niah_none_fixed']} | "
                    f"{100*d['cwe_recall']:.1f}% | {100*d['cwe_easy_recall']:.1f}% |\n")
        if "l0_vs_others" in combined:
            x = combined["l0_vs_others"]
            f.write(f"\nl0 vs others: {x['diff_pp']:+.1f}pp, z={x['z']:+.2f}, "
                    f"significant={x['significant_0.05']}\n")
    print(f"wrote {out}\nwrote {md}")


if __name__ == "__main__":
    main()
