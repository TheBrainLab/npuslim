#!/usr/bin/env python3
"""M0.3 全量语料标注与内容去重：clean/*.jsonl -> corpus_index.jsonl。

每样本一行：对齐键 (file, line_no)（id 非唯一，仅作 task 分组键，见 M0.1 结论）；
内容 hash 用于 exact-dup 标记（保留首次出现，后续标 is_dup，不删行——logits 按
行号对齐，采样时排除即可）；token 估计 = bytes × tokens_per_byte（全局校准，
M4 前用服务器 index.jsonl 的精确 n_tokens 复核）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

TOK_PER_BYTE = 0.197  # §12.9: 561,684,804 tok / 2.85 GB，Qwen3 tokenizer 全局实测


def parse_domain(stem: str) -> tuple[str, str, str]:
    parts = stem.split("-", 2)
    bucket = parts[0]
    category = parts[1] if len(parts) > 1 else ""
    source = parts[2] if len(parts) > 2 else ""
    return bucket, category, source


def content_hash(rec: dict) -> str:
    """对 messages 做规范化 hash（exact-dup 检测）。"""
    norm = json.dumps(rec.get("messages"), ensure_ascii=False, sort_keys=True)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.corpus_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERR] no jsonl under {args.corpus_dir}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hash_seen: dict[str, str] = {}  # content_hash -> 首次出现 sample_key
    stats: Counter[str] = Counter()
    n_total = 0

    with args.out.open("w", encoding="utf-8") as out:
        for f in files:
            bucket, category, source = parse_domain(f.stem)
            with f.open(encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh):
                    rec = json.loads(raw)
                    n_bytes = len(raw.encode("utf-8"))
                    ch = content_hash(rec)
                    key = f"{f.name}:{line_no}"
                    is_dup = ch in hash_seen
                    if is_dup:
                        stats["dup_skipped"] += 1
                    else:
                        hash_seen[ch] = key
                    out.write(json.dumps({
                        "sample_key": key,
                        "file": f.name,
                        "line_no": line_no,
                        "id": rec.get("id"),
                        "bucket": bucket,
                        "category": category,
                        "source": source,
                        "n_bytes": n_bytes,
                        "token_est": round(n_bytes * TOK_PER_BYTE),
                        "content_hash": ch,
                        "is_dup": is_dup,
                    }, ensure_ascii=False) + "\n")
                    stats[f"bucket_{bucket}"] += 1
                    stats["tokens_total"] += round(n_bytes * TOK_PER_BYTE)
                    n_total += 1

    # 汇总报告（去重后 token 按桶统计需二次扫描；这里输出总量与 dup 数即可，
    # 桶级 token 在 sample_trainset.py 里基于 corpus_index 精确统计）
    report = {
        "n_samples": n_total,
        "tokens_est_total": stats.pop("tokens_total", 0),
        "dup_skipped": stats.pop("dup_skipped", 0),
        "per_bucket_samples": dict(sorted(stats.items())),
        "tok_per_byte": TOK_PER_BYTE,
    }
    side = args.out.with_suffix(".stats.json")
    side.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] index -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
