#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_stats.py — 四维评测统计工具（1010，P0-3/P0-5）

背景（W4A8_总结与规划0908.md §7.5）：历史对照数字曾因人工转抄出现列错位（坑㊿），
且判定长期未带置信区间。本工具把"对照表由脚本从实测 json 生成"制度化，并提供
逐题 paired 检验（McNemar + paired bootstrap）——比独立 SE 的功效高。

子命令：
  table   从 lm-eval results_*.json 扫描 产物×轮次（目录名 <artifact>_rN/<hash>/），
          输出四维均值/波动/CI 与 artifact 间配对差值表（unpaired SE_Δ）。
  paired  从两个 --log_samples 目录的 samples_*.jsonl 逐题配对（join key = doc_hash），
          计算 McNemar 精确检验 + paired bootstrap CI。

用法示例：
  python3 eval_stats.py table --root <audit1009> \
      --pairs "p2clip_s1234:w4a8c8,l0_clip_lo05_omni:w4a8c8" --out table.md
  python3 eval_stats.py paired --a <dirA> --b <dirB> --task gsm8k --boot 10000

仅依赖 stdlib。兼容 lm-eval 0.4.x 的 results/samples 结构（audit1009 实测：
wikitext=word_perplexity,none；gsm8k=exact_match,strict-match；mmlu=acc,none；
humaneval=pass@1,create_test）。
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

# task -> (首选 metric 全键列表, higher_is_better, stderr 全键候选列表)
METRIC_MAP = {
    "wikitext": (["word_perplexity,none"], False, []),
    "gsm8k": (["exact_match,strict-match"], True, ["exact_match_stderr,strict-match"]),
    "mmlu": (["acc,none"], True, ["acc_stderr,none"]),
    "humaneval": (["pass@1,create_test", "pass@1,none"], True,
                  ["pass@1_stderr,create_test", "pass@1_stderr,none"]),
}

TASK_SPLIT_RE = re.compile(r"^(?P<base>mmlu|gsm8k|wikitext|humaneval)")
ART_ROUND_RE = re.compile(r"^(?P<art>.+)_(?P<rnd>r\d+)$")

# ---------------------------------------------------------------- table 模式


def parse_results_json(path):
    """解析单个 lm-eval results json → {task_base: (value, stderr, n, higher)}"""
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
            continue  # 只取聚合行，不取各 subject 行
        spec = METRIC_MAP.get(base)
        if not spec:
            continue
        keys, higher, se_keys = spec
        val = None
        for k in keys:  # 全键精确匹配优先
            if k in metrics and isinstance(metrics[k], (int, float)):
                val = metrics[k]
                break
        if val is None:
            continue
        se = None
        for sk in se_keys:
            if sk in metrics and isinstance(metrics[sk], (int, float)):
                se = metrics[sk]
                break
        n = metrics.get("sample_len")
        if n is None:
            n = ns.get(task, {}).get("effective")
        out[base] = (float(val), (float(se) if se is not None else None),
                     int(n) if n else None, higher)
    return out


def split_tag(tag):
    m = ART_ROUND_RE.match(tag)
    return (m.group("art"), m.group("rnd")) if m else (tag, "r0")


def collect_runs(root):
    """root/<artifact>_rN/<hash>/results_*.json → {artifact: {round: {task: (...)}}}"""
    data = defaultdict(dict)
    if not os.path.isdir(root):
        return data
    for tag in sorted(os.listdir(root)):
        tag_dir = os.path.join(root, tag)
        if not os.path.isdir(tag_dir):
            continue  # 跳过 .log/.list 等文件
        art, rnd = split_tag(tag)
        for rj in sorted(glob.glob(os.path.join(tag_dir, "*", "results_*.json"))):
            parsed = parse_results_json(rj)
            if parsed and rnd not in data[art]:  # 同轮多文件取首个（均为同轮产物）
                data[art][rnd] = parsed
    return data


def binom_se(p, n):
    if not n:
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def cmd_table(args):
    data = collect_runs(args.root)
    if not data:
        print("no results found under", args.root)
        return 1
    lines = ["| artifact | metric | runs | per-round values | mean | spread(max-min) | per-run SE | n |",
             "|---|---|---|---|---|---|---|---|"]
    agg = defaultdict(dict)  # artifact -> metric -> (mean, n, higher)
    for art in sorted(data):
        per_metric = defaultdict(list)
        meta = {}
        for rnd in sorted(data[art]):
            for task, (v, se, n, higher) in data[art][rnd].items():
                per_metric[task].append(v)
                meta[task] = (se, n, higher)
        for task, vals in sorted(per_metric.items()):
            mean = sum(vals) / len(vals)
            spread = max(vals) - min(vals)
            se, n, higher = meta[task]
            per_run_se = se if se is not None else (binom_se(mean, n) if n else None)
            agg[art][task] = (mean, n, higher)
            se_s = f"{per_run_se*100:.3f}pp" if per_run_se else "—(ppl 无闭式)"
            lines.append(
                f"| {art} | {task} | {len(vals)} | {', '.join(f'{v:.4f}' for v in vals)} "
                f"| {mean:.4f} | {spread:.4f} | {se_s} | {n or '—'} |")
    if args.pairs:
        lines += ["", "## artifact 间配对差值（unpaired SE_Δ=√(SEa²+SEb²)；|Δ|<2σ 视为噪声内）", "",
                  "| pair | metric | per-round Δ | Δmean | SE_Δ | σ | 判定 |", "|---|---|---|---|---|---|---|"]
        for pair in [x for x in args.pairs.split(",") if x]:
            a, b = pair.split(":")
            ta, tb = agg.get(a), agg.get(b)
            if not ta or not tb:
                continue
            rounds = sorted(set(data[a]) & set(data[b]))
            for task in sorted(set(ta) & set(tb)):
                deltas = []
                for r in rounds:
                    if task in data[a][r] and task in data[b][r]:
                        deltas.append(data[a][r][task][0] - data[b][r][task][0])
                if not deltas:
                    continue
                mean = sum(deltas) / len(deltas)
                (ma, na, _), (mb, nb, _) = ta[task], tb[task]
                sea = binom_se(ma, na) or 0.0
                seb = binom_se(mb, nb) or 0.0
                se_d = math.hypot(sea, seb)
                sigma = (mean / se_d) if se_d else 0.0
                verdict = "噪声内" if abs(sigma) < 2 else "超 2σ"
                lines.append(
                    f"| {a} vs {b} | {task} | {', '.join(f'{x:+.4f}' for x in deltas)} "
                    f"| {mean:+.4f} | {se_d:.4f} | {sigma:.1f}σ | {verdict} |")
    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwritten -> {args.out}")
    return 0


# ---------------------------------------------------------------- paired 模式


def load_samples(dirpath, task, metric_field):
    """dir 下 samples_<task>*.jsonl（mmlu=按学科多文件）→ {doc_hash: 0/1}"""
    if task == "mmlu":
        files = sorted(glob.glob(os.path.join(dirpath, "samples_mmlu_*.jsonl")))
    else:
        files = sorted(glob.glob(os.path.join(dirpath, f"samples_{task}*.jsonl")))
    out = {}
    dup = 0
    base_field = metric_field.split(",")[0]
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
                        if k.startswith(base_field) and isinstance(val, (int, float)):
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
    """discordant: b01=A对B错, b10=A错B对；双侧精确二项检验"""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def paired_bootstrap(a, b, boot, seed):
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if not n:
        return None
    rng = random.Random(seed)
    a_vals = [a[k] for k in keys]
    b_vals = [b[k] for k in keys]
    diffs = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_vals[j] for j in idx) / n
        db = sum(b_vals[j] for j in idx) / n
        diffs.append(db - da)
    diffs.sort()
    lo = diffs[int(0.025 * boot)]
    hi = diffs[min(boot - 1, int(0.975 * boot))]
    frac_pos = sum(1 for d in diffs if d > 0) / boot
    return lo, hi, frac_pos


def cmd_paired(args):
    field = args.metric_field
    if field is None:
        field = {"gsm8k": "exact_match", "mmlu": "acc", "humaneval": "pass@1",
                 "wikitext": "word_perplexity"}.get(args.task, "acc")
    a, fa, dupa = load_samples(args.a, args.task, field)
    b, fb, dupb = load_samples(args.b, args.task, field)
    if not a or not b:
        print(f"samples not found: a n={len(a)} files={fa}, b n={len(b)} files={fb} "
              f"(field={field}; 需要 --log_samples 产物)")
        return 1
    joined = sorted(set(a) & set(b))
    acc_a = sum(a[k] for k in joined) / len(joined)
    acc_b = sum(b[k] for k in joined) / len(joined)
    n10 = sum(1 for k in joined if a[k] == 1 and b[k] == 0)  # A 对 B 错
    n01 = sum(1 for k in joined if a[k] == 0 and b[k] == 1)  # A 错 B 对
    p = mcnemar_exact(n10, n01)
    bs = paired_bootstrap(a, b, args.boot, args.seed)
    print(f"== paired: A={args.a}  B={args.b} | task={args.task} field={field}")
    print(f"joined n={len(joined)} (dup skipped: a={dupa}, b={dupb})")
    print(f"acc A={acc_a:.4f}  acc B={acc_b:.4f}  Δ(B-A)={(acc_b-acc_a)*100:+.2f}pp")
    print(f"discordant: A对B错={n10}  A错B对={n01}  -> McNemar exact p={p:.4g}")
    if bs:
        lo, hi, frac_pos = bs
        print(f"paired bootstrap (B={args.boot}, seed={args.seed}): "
              f"Δ 95%CI=[{lo*100:+.2f}pp, {hi*100:+.2f}pp], P(Δ>0)={frac_pos:.3f}")
    verdict = "显著差异（p<0.05）" if p < 0.05 else "统计不可区分（paired）"
    print(f"verdict: {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"paired A={args.a} B={args.b} task={args.task}\n"
                    f"joined_n={len(joined)} accA={acc_a:.4f} accB={acc_b:.4f} "
                    f"d={(acc_b-acc_a)*100:+.2f}pp n10={n10} n01={n01} "
                    f"mcnemar_p={p:.4g} bootCI=[{lo*100:+.2f},{hi*100:+.2f}]pp\n"
                    f"verdict: {verdict}\n")
    return 0


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="四维评测统计工具（CI + paired 检验）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="多产物多轮聚合表 + artifact 间差值")
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
