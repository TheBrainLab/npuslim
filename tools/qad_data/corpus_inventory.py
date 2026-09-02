#!/usr/bin/env python3
"""M0.1 QAD 语料盘点：逐文件样本数 / 大小 / 域标签 / id 唯一性，对账 489,148。

对齐关系：本语料（clean/*.jsonl 9 文件）即 teacher logits 缓存（§12.9，845GB top-100
npz）的输入语料，样本级对齐键 = (file, id)。index.jsonl 在服务器上，本机用行数
对账（expected_total）代替逐样本对账，shard 命中校验推迟到 M4 前置检查。

用法:
    python tools/qad_data/corpus_inventory.py \
        --corpus-dir workspace/glm52-qad/glm52-qad-corpus/processed/corpus_v1_2026-08-27/clean \
        --out workspace/glm52-qad/qad_data/corpus_stats.json
    # 快速模式（只数行数，不扫 id 唯一性）: --fast
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_TOTAL = 489_148  # 主文档 §12.9 实测值
EXPECTED_SCHEMA = {"id", "split", "scenario", "messages", "chunk_plan", "labels", "meta"}


def parse_domain(stem: str) -> tuple[str, str, str]:
    """文件名 A-long_horizon-tau-bench -> (bucket, category, source)。"""
    parts = stem.split("-", 2)
    bucket = parts[0]
    category = parts[1] if len(parts) > 1 else ""
    source = parts[2] if len(parts) > 2 else ""
    return bucket, category, source


def scan_file(path: Path, fast: bool) -> dict:
    n_lines = 0
    n_bytes = 0
    n_bad_json = 0
    id_seen: set[str] | None = None if fast else set()
    n_dup_id = 0
    scenarios: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    schema_ok = True

    with path.open("rb") as fh:
        for raw in fh:
            n_lines += 1
            n_bytes += len(raw)
            if fast and scenarios and n_lines % 1000 == 0:
                continue  # fast 模式下不逐行 parse（首行 schema 已查）
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            if n_lines == 1:
                schema_ok = EXPECTED_SCHEMA.issubset(rec.keys())
            if not fast:
                sid = rec.get("id")
                if sid is not None:
                    if sid in id_seen:
                        n_dup_id += 1
                    else:
                        id_seen.add(sid)
            scenarios[str(rec.get("scenario"))] += 1
            splits[str(rec.get("split"))] += 1

    return {
        "n_samples": n_lines,
        "n_bytes": n_bytes,
        "avg_bytes_per_sample": round(n_bytes / max(n_lines, 1), 1),
        "n_bad_json": n_bad_json,
        "n_unique_id": (None if fast else len(id_seen) if id_seen else 0),
        "n_dup_id": n_dup_id,
        "schema_ok": schema_ok,
        "scenarios": dict(scenarios),
        "splits": dict(splits),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fast", action="store_true", help="只数行数，不扫描 id 唯一性")
    args = ap.parse_args()

    files = sorted(args.corpus_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERR] no jsonl under {args.corpus_dir}", file=sys.stderr)
        return 2

    entries = []
    total = 0
    total_bytes = 0
    ok = True
    for f in files:
        bucket, category, source = parse_domain(f.stem)
        st = scan_file(f, args.fast)
        st.update({"file": f.name, "bucket": bucket, "category": category, "source": source})
        # 完整性检查：坏行 / 重复 id / schema 缺字段
        if st["n_bad_json"] or (not args.fast and st["n_dup_id"]) or not st["schema_ok"]:
            ok = False
        entries.append(st)
        total += st["n_samples"]
        total_bytes += st["n_bytes"]
        print(
            f"{f.name:48s} {st['n_samples']:>7d}  {st['n_bytes'] / 1e6:>9.1f} MB"
            f"  {st['avg_bytes_per_sample']:>9.1f} B/条"
            + ("" if args.fast else f"  dup_id={st['n_dup_id']}")
        )

    recon = "OK" if total == EXPECTED_TOTAL else "MISMATCH"
    if recon != "OK":
        ok = False
    print("-" * 96)
    print(f"TOTAL {total} / expected {EXPECTED_TOTAL}  [{recon}]   {total_bytes / 1e9:.2f} GB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "expected_total": EXPECTED_TOTAL,
        "actual_total": total,
        "reconciliation": recon,
        "total_bytes": total_bytes,
        "fast_mode": args.fast,
        "files": entries,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
