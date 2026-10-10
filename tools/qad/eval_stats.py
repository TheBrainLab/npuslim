#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_stats.py — 四维评测统计工具（1010，P0-3/P0-5）

背景（W4A8_总结与规划0908.md §7.5）：历史对照数字曾因人工转抄出现列错位（坑㊿），
且判定长期未带置信区间。本工具把"对照表由脚本从实测 json 生成"制度化，并提供
逐题 paired 检验（McNemar + paired bootstrap）——比独立 SE 的功效高。

子命令：
  table   从 lm-eval results_*.json 扫描产物×轮次，输出四维均值/波动/CI 与配对差值表。
  paired  从两个 --log_samples 目录的 samples_*.jsonl 逐题配对（join key = doc_hash），
          计算 McNemar 精确检验 + paired bootstrap CI。

用法示例：
  python3 eval_stats.py table --root <audit1009> \
      --pairs "p2clip_s1234:w4a8c8,l0_clip_lo05_omni:w4a8c8"
  python3 eval_stats.py paired --a <dirA> --b <dirB> --task gsm8k --boot 10000

仅依赖 stdlib。兼容 lm-eval 0.4.x 的 results/samples 结构（audit1009 实测）。
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------- 常量
# task -> (metric_key, higher_is_better, stderr_key)
METRIC_MAP = {
    "wikitext": ("word_perplexity,none", False, None),
    "gsm8k": ("exact_match,strict-match", True, "exact_match_stderr,strict-match"),
    "gsm8k_flex": ("exact_match,flexible-extract", True, "exact_match_stderr,flexible-extract"),
    "mmlu": ("acc,none", True, "acc_stderr,none"),
    "humaneval": ("pass@1,none", True, None),
}

TASK_SPLIT_RE = re.compile(r"^(?P<base>mmlu|gsm8k|wikitext|humaneval)")

# ---------------------------------------------------------------- table 模式


def parse_results_json(path):
    """解析单个 lm-eval results json → {task_base: (value, stderr, n)}"""
    with open(path) as f:
        d = json.load(f)
    out = {}
    results = d.get("results", {})
    ns = d.get("n-samples", {})
    for task, metrics in results.items():
        m = TASK_SPLIT_RE.match(task)
        if not m:
            continue
        base = m.group("base")
        if base == "mmlu" and task != "mmlu":
            continue  # 只取聚合行，不取各 subject
        mk = METRIC_MAP.get(base)
        if not mk:
            continue
        key, higher, se_key = mk
        for k, v in metrics.items():
            if k.startswith(key.split(",")[0]) and isinstance(v, (int, float)):
                # 精确匹配 "key" 形如 'exact_match,strict-match' / 'acc,none'
                if k == key or k.startswith(key):
                    se = metrics.get(se_key) if se_key else None
                    n = metrics.get("sample_len")
                    if n is None:
                        n = ns.get(task, {}).get("effective")
                    out[base] = (float(v), (float(se) if isinstance(se, (int, float)) else None),
                                 int(n) if n else None, higher)
                    break
    return out


def collect_runs(root):
    """root/<tag>/<hash>/results_*.json → {tag: {round_or_hash: {task: (v, se, n, higher)}}}"""
    data = defaultdict(dict)
    if not os.path.isdir(root):
        return data
    for tag in sorted(os.listdir(root)):
        tag_dir = os.path.join(root, tag)
        if not os.path.isdir(tag_dir):
            continue  # 跳过 .log/.list 文件
        for rj in glob.glob(os.path.join(tag_dir, "*", "results_*.jsonl")) + \
                 glob.glob(os.path.join(tag_dir, "*", "results_*.json")):
            rid = os.path.basename(rj).replace("results_", "")[:16]
            parsed = parse_results_json(rj)
            if parsed:
                data[tag][rid] = parsed
    return data


def binom_se(p, n):
    if not n:
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def agg(values):
    mean = sum(values) / len(values)
    return mean, (max(values) - min(values))


def cmd_table(args):
    data = collect_runs(args.root)
    if not data:
        print("no results found under", args.root)
        return 1
    lines = ["| artifact | metric | runs | values | mean | spread | per-run SE | n |",
             "|---|---|---|---|---|---|---|---|"]
    agg_by_artifact = defaultdict(dict)  # artifact -> metric -> (mean, n)
    for tag in sorted(data):
        per_metric = defaultdict(list)
        meta = {}
        for rid, tasks in data[tag].items():
            for task, (v, se, n, higher) in tasks.items():
                per_metric[task].append(v)
                meta[task] = (se, n, higher)
        for task, vals in sorted(per_metric.items()):
            mean, spread = agg(vals)
            se, n, higher = meta[task]
            per_run_se = se if se is not None else (binom_se(mean, n) if n else None)
            agg_by_artifact[tag][task] = (mean, n, higher)
            se_s = f"{per_run_se*100:.3f}pp" if per_run_se else "—"
            lines.append(
                f"| {tag} | {task} | {len(vals)} | {', '.join(f'{v:.4f}' for v in vals)} "
                f"| {mean:.4f} | {spread:.4f} | {se_s} | {n or '—'} |")
    # 配对差值（unpaired SE of difference）
    if args.pairs:
        lines += ["", "## 配对差值（unpaired SE_Δ = √(SEa²+SEb²)；差值 < 2σ 视为噪声内）", "",
                  "| pair | metric | Δ(rounds) | Δmean | SE_Δ | σ | 判定 |", "|---|---|---|---|---|---|---|"]
        for pair in args.pairs.split(","):
            a, b = pair.split(":")
            ta, tb = agg_by_artifact.get(a), agg_by_artifact.get(b)
            if not ta or not tb:
                continue
            for task in sorted(set(ta) & set(tb)):
                # 用每轮值配对求 delta 序列
                da = [v for rid in sorted(data[a]) if task in data[a][rid] for v in [data[a][rid][task][0]]]
                db = [v for rid in sorted(data[b]) if task in data[b][rid] for v in [data[b][rid][task][0]]]
                if len(da) != len(db):
                    continue
                deltas = [x - y for x, y in zip(da, db)]
                mean, spread = agg(deltas)
                (ma, na, ha), (mb, nb, hb) = ta[task], tb[task]
                sea = binom_se(ma, na) or 0.0
                seb = binom_se(mb, nb) or 0.0
                se_d = math.hypot(sea, seb)
                sigma = (mean / se_d) if se_d else 0.0
                verdict = "噪声内" if abs(sigma) < 2 else ("显著" if abs(sigma) >= 2 else "边界")
                lines.append(
                    f"| {a} vs {b} | {task} | {', '.join(f'{x:+.4f}' for x in deltas)} | {mean:+.4f} "
                    f"| {se_d:.4f} | {sigma:.1f}σ | {verdict} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwritten -> {args.out}")
    return 0


# ---------------------------------------------------------------- paired 模式


def load_samples(dirpath, task, metric_field):
    """dir 下 samples_<task>*.jsonl → {doc_hash: 0/1}；mmlu 按学科多文件合并"""
    pattern = os.path.join(dirpath, f"samples_{task}*.jsonl") if task != "mmlu" \
        else os.path.join(dirpath, "samples_mmlu_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files and task != "mmlu":  # mmlu 聚合名兜底
        files = sorted(glob.glob(os.path.join(dirpath, f"samples_{task}_*.jsonl")))
    out = {}
    dup = 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                h = d.get("doc_hash")
                if h is None:
                    h = f"{d.get('doc_id')}"
                v = d.get(metric_field)
                if v is None:
                    for k, val in d.items():
                        if k.startswith(metric_field.split(",")[0]) and isinstance(val, (int, float)):
                            v = val
                            break
                if v is None:
                    continue
                if h in out:
                    dup += 1
                    continue
                out[h] = 1.0 if float(v) >= 0.5 else 0.0
    return out, len(files), dup


def mcnemar_exact(b01, b10):
    """discordant: b01 = a对b错, b10 = a错b对；双侧精确二项检验"""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def paired_bootstrap(a, b, boot, seed):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    rng = random.Random(seed)
    n = len(keys)
    diffs = []
    a_vals = [a[k] for k in keys]
    b_vals = [b[k] for k in keys]
    base = (sum(a_vals) / n, sum(b_vals) / n)
    for _ in range(boot):
        i = [rng.randrange(n) for _ in range(n)]
        da = sum(a_vals[j] for j in i) / n
        db = sum(b_vals[j] for j in i) / n
        diffs.append(db - da)
    diffs.sort()
    lo = diffs[int(0.025 * boot)]
    hi = diffs[min(boot - 1, int(0.975 * boot))]
    frac_pos = sum(1 for d in diffs if d > 0) / boot
    return keys, base, lo, hi, frac_pos


def cmd_paired(args):
    field = args.metric_field
    if field is None:
        field = {"gsm8k": "exact_match", "mmlu": "acc", "humaneval": "pass@1",
                 "wikitext": "word_perplexity"}.get(args.task, "acc")
    a, fa, dupa = load_samples(args.a, args.task, field)
    b, fb, dupb = load_samples(args.b, args.task, field)
    if not a or not b:
        print(f"samples not found: a={len(a)} files={fa}, b={len(b)} files={fb} "
              f"(field={field}; 需要 --log_samples 产物)")
        return 1
    joined = sorted(set(a) & set(b))
    acc_a = sum(a[k] for k in joined) / len(joined)
    acc_b = sum(b[k] for k in joined) / len(joined)
    n10 = sum(1 for k in joined if a[k] == 1 and b[k] == 0)  # a 对 b 错
    n01 = sum(1 for k in joined if a[k] == 0 and b[k] == 1)  # a 错 b 对
    p = mcnemar_exact(n10, n01)
    bs = paired_bootstrap(a, b, args.boot, args.seed)
    disc = sorted(set(a) & set(b))[:0]
    print(f"== paired: {args.a} (A) vs {args.b} (B) | task={args.task} field={field}")
    print(f"joined n={len(joined)} (dup skipped: a={dupa}, b={dupb})")
    print(f"acc A={acc_a:.4f}  acc B={acc_b:.4f}  Δ(B-A)={(acc_b-acc_a)*100:+.2f}pp")
    print(f"discordant: A对B错={n10}  A错B对={n01}  -> McNemar exact p={p:.4g}")
    if bs:
        _, base, lo, hi, frac_pos = bs
        print(f"paired bootstrap (B={args.boot}, seed={args.seed}): "
              f"Δ 95%CI=[{lo*100:+.2f}pp, {hi*100:+.2f}pp], P(Δ>0)={frac_pos:.3f}")
    verdict = "显著差异" if p < 0.05 else "统计不可区分（paired）"
    print(f"verdict: {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"paired {args.a} vs {args.b} task={args.task}: "
                    f"accA={acc_a:.4f} accB={acc_b:.4f} d={(acc_b-acc_a)*100:+.2f}pp "
                    f"n10={n10} n01={n01} mcnemar_p={p:.4g} bootCI=[{lo*100:+.2f},{hi*100:+.2f}]pp {verdict}\n")
    return 0


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="四维评测统计工具（CI + paired 检验）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="多产物多轮聚合表 + unpaired 差值")
    t.add_argument("--root", required=True)
    t.add_argument("--pairs", default="")
    t.add_argument("--out", default="")
    p = sub.add_parser("paired", help="两目录逐题配对（McNemar + bootstrap）")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--task", required=True, choices=["gsm8k", "mmlu", "humaneval", "wikitext"])
    p.add_argument("--metric-field", default=None)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "table":
        sys.exit(cmd_table(args))
    else:
        sys.exit(cmd_paired(args))


if __name__ == "__main__":
    main()
