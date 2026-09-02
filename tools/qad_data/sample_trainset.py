#!/usr/bin/env python3
"""M0.4/M0.5 按配比组装 QAD 训练集：corpus_index.jsonl + mix_config.yaml -> qad_train_v1.jsonl。

流程：过滤内容重复 -> 按 (file,id) 分组（防泄漏）-> 桶内组级洗牌（seed）->
贪心填充至桶目标 token -> 输出训练集与配比对账报告。
sample_key 全部来自本地 489k 语料；shard 命中校验（vs 服务器 index.jsonl）留待 M4 前置检查。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-index", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mix: dict[str, float] = cfg["mix"]
    samp = cfg["sampling"]
    tok_per_byte = cfg["token_estimate"]["tokens_per_byte"]
    target_total = int(samp["target_tokens"])
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        print(f"[ERR] mix weights sum to {sum(mix.values()):.4f}, expect 1.0", file=sys.stderr)
        return 2

    # 读 index：过滤 dup，按 (file,id) 分组
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_scanned = n_dups = 0
    bucket_tokens_avail: Counter[str] = Counter()
    with args.corpus_index.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n_scanned += 1
            if samp.get("exclude_content_dups", True) and r["is_dup"]:
                n_dups += 1
                continue
            groups[(r["file"], r["id"])].append(r)
            bucket_tokens_avail[r["category"]] += r["token_est"]

    # 组级元数据：组内 token 与桶（同 id 组内桶必相同——id 唯一属于一个文件）
    group_list: list[tuple[str, int, list[dict]]] = []  # (bucket, group_tokens, members)
    for (_f, _id), members in groups.items():
        group_list.append((members[0]["category"], sum(m["token_est"] for m in members), members))

    rng = random.Random(samp["seed"])
    buckets: dict[str, list] = defaultdict(list)
    for g in group_list:
        buckets[g[0]].append(g)
    for b in buckets.values():
        rng.shuffle(b)

    # 贪心填充：每桶目标 = target_total × mix[b]，组级整取
    picked: list[list[dict]] = []
    picked_tokens: Counter[str] = Counter()
    report_buckets = {}
    for bucket, weight in mix.items():
        target = int(target_total * weight)
        taken = 0
        n_groups = 0
        for gtok, members in ((g[1], g[2]) for g in buckets.get(bucket, [])):
            if taken >= target:
                break
            picked.append(members)
            taken += gtok
            n_groups += 1
        picked_tokens[bucket] = taken
        short = "  [POOL EXHAUSTED]" if taken < target else ""
        report_buckets[bucket] = {
            "target_tokens": target, "picked_tokens": taken,
            "avail_tokens": bucket_tokens_avail[bucket],
            "groups": n_groups, "share_actual": round(taken / max(target_total, 1), 4),
        }
        print(f"{bucket:16s} target={target:>12,d}  picked={taken:>12,d}  "
              f"groups={n_groups:>6,d}{short}")

    total = sum(picked_tokens.values())
    for b in report_buckets:
        report_buckets[b]["share_actual"] = round(picked_tokens[b] / max(total, 1), 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for members in picked:
            for m in members:
                out.write(json.dumps({
                    "sample_key": m["sample_key"],
                    "id": m["id"],
                    "category": m["category"],
                    "source": m["source"],
                    "token_est": m["token_est"],
                    "n_bytes": m["n_bytes"],
                }, ensure_ascii=False) + "\n")

    report = {
        "n_scanned": n_scanned, "n_dups_excluded": n_dups,
        "n_samples": sum(len(m) for m in picked),
        "total_tokens_est": total,
        "target_tokens": target_total,
        "tok_per_byte": tok_per_byte,
        "seed": samp["seed"],
        "buckets": report_buckets,
    }
    side = args.out.with_suffix(".report.json")
    side.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"TOTAL {report['n_samples']:,} samples, {total:,} tok-est "
          f"(target {target_total:,})")
    print(f"[OK] trainset -> {args.out}")
    print(f"[OK] report  -> {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
