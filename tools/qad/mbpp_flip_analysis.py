#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mbpp 翻转集题型归因（1011）：p2clip fail 且 tgt pass 的题簇分类（零 NPU 离线）。
输入 samples_mbpp_*.jsonl 目录（含 per-sample pass_at_1 + doc.text/doc.test_list/doc.task_id）。
"""
import argparse
import collections
import glob
import json
import os
import re

KEY_PAT = [
    ("string", re.compile(r"(string|character|char|letter|word|sentence| theoretically|substring|capitalize|palindrome|切割|字符串)", re.I)),
    ("list", re.compile(r"(list|tuple|array|elements|elements of|list中|元素)", re.I)),
    ("math", re.compile(r"(number|integer|digit|modulo|factorial|prime|sum|average|square|cube|number of|计算|求)", re.I)),
    ("dict/set", re.compile(r"(dictionary|dict|set|frequency|count of|counts|duplicates)", re.I)),
    ("matrix/2d", re.compile(r"(matrix|2d|nested|rows|columns)", re.I)),
    ("sort/search", re.compile(r"(sort|search|find the|index of|swap|min|max|split)", re.I)),
    ("file/conversion", re.compile(r"(file|json|csv|xml|convert|format|parse)", re.I)),
]


def classify(text: str) -> str:
    for name, pat in KEY_PAT:
        if pat.search(text):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dir with samples_mbpp jsonl (A, e.g. p2clip)")
    ap.add_argument("--b", required=True, help="dir with samples_mbpp jsonl (B, e.g. tgt)")
    args = ap.parse_args()

    def load(d):
        fs = glob.glob(os.path.join(d, "**", "samples_mbpp_*.jsonl"), recursive=True) or \
             glob.glob(os.path.join(d, "samples_mbpp_*.jsonl"))
        recs = {}
        for fp in fs:
            for line in open(fp):
                r = json.loads(line)
                recs[r["doc_hash"]] = r
        return recs

    A, B = load(args.a), load(args.b)
    keys = [k for k in A if k in B]
    pfail_bok = [k for k in keys if A[k]["pass_at_1"] == 0 and B[k]["pass_at_1"] == 1]
    both_fail = [k for k in keys if A[k]["pass_at_1"] == 0 and B[k]["pass_at_1"] == 0]
    print(f"joined n={len(keys)}  A-fail&B-pass={len(pfail_bok)}  both-fail={len(both_fail)}")
    # 分布统计
    cls = collections.Counter(classify(A[k]["doc"]["text"]) for k in pfail_bok)
    tot = collections.Counter(classify(A[k]["doc"]["text"]) for k in keys)
    print("\n== A-fail&B-pass 翻转集的题型分布（文档题目分词聚类） ==")
    for name, cnt in cls.most_common():
        print(f"  {name:>12}: {cnt:4d}/{tot[name]:4d} ({cnt/max(tot[name],1)*100:5.1f}%)")
    # 参照：both-fail 中同分类占比
    print("\n== 双方都 fail 的题型分布（collateral 参考） ==")
    c2 = collections.Counter(classify(A[k]["doc"]["text"]) for k in both_fail)
    t2 = collections.Counter(classify(A[k]["doc"]["text"]) for k in keys)
    for name, cnt in c2.most_common():
        print(f"  {name:>12}: {cnt:4d}/{t2[name]:4d} ({cnt/max(t2[name],1)*100:5.1f}%)")
    # save flip 样本
    out = os.path.join(args.a, "flip_analysis.json")
    flips = [{"task_id": A[k]["doc"]["task_id"], "text": A[k]["doc"]["text"][:80],
              "cat": classify(A[k]["doc"]["text"])} for k in pfail_bok]
    json.dump({"joined": len(keys), "p2clip_only_fail": len(flips), "both_fail": len(both_fail),
               "flip_categories": dict(cls), "bothfail_categories": dict(c2), "flips": flips},
              open(out, "w"), indent=1, ensure_ascii=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
