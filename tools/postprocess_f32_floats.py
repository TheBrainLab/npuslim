#!/usr/bin/env python3
"""Post-process a quantized product: cast FLOAT norm weights + lm_head BF16 -> F32.

E7 experiment (W4A8_总结与规划0908.md §1.7): strictly align the FLOAT storage
dtype of the modelslim w4a8c8 product (W4A8_修改总结.md §18.8): 159 norm weights
(79x input_layernorm + 79x post_attention_layernorm + 1x model.norm) + lm_head
stored as F32 instead of BF16. Zero-quant-cost test of §1.5 gap ②③.

What it does:
  1. Read model.safetensors.index.json, enumerate target tensors, hard-assert
     counts (159 norms + 1 lm_head) so silent drift is caught.
  2. Copy every non-shard file as-is (config/tokenizer/description/index/...).
     Assert quant_model_description.json FLOAT entries carry NO dtype field
     (verified 2026-09-09: plain "FLOAT" strings) -> nothing to sync.
  3. Rewrite shard files: shards without targets are byte-copied; shards with
     targets are loaded, targets cast to F32, saved with original metadata.
  4. Re-scan all shard headers: targets are F32, non-targets byte-identical
     dtypes/shapes vs source. Write a manifest json.

Source product is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

NORM_PATTERNS = (
    re.compile(r"^model\.layers\.\d+\.input_layernorm\.weight$"),
    re.compile(r"^model\.layers\.\d+\.post_attention_layernorm\.weight$"),
    re.compile(r"^model\.norm\.weight$"),
)
LM_HEAD = "lm_head.weight"
EXPECT_NORMS = 159  # 79 + 79 + 1, per w4a8c8 (修改总结 §18.8)


def is_target(name: str) -> bool:
    if name == LM_HEAD:
        return True
    return any(p.match(name) for p in NORM_PATTERNS)


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_targets(weight_map: dict) -> dict[str, str]:
    """Return {tensor_name: shard_file} for all target tensors."""
    targets = {}
    norm_count = 0
    lm_count = 0
    for name, shard in weight_map.items():
        if not is_target(name):
            continue
        targets[name] = shard
        if name == LM_HEAD:
            lm_count += 1
        else:
            norm_count += 1
    if norm_count != EXPECT_NORMS:
        raise SystemExit(
            f"[FATAL] norm target count {norm_count} != {EXPECT_NORMS} "
            f"(w4a8c8 对齐集合漂移，拒绝执行)"
        )
    if lm_count != 1:
        raise SystemExit(f"[FATAL] lm_head count {lm_count} != 1")
    return targets


def assert_description_no_dtype(src: str) -> None:
    """quant_model_description.json 的张量条目必须不带 dtype 字段（纯字符串）。

    张量条目 = 键含 "." 的项（如 model.layers.0.input_layernorm.weight: FLOAT）。
    顶层标量元数据（version / model_quant_type / group_size 等）不受约束。
    若上游把张量条目改成结构化 dict（如 {"dtypes": ["FLOAT","F32"]}），E7 需
    同步该字段；当前实现为纯字符串，无需同步 —— 此断言把"无需同步"变成受检事实。
    """
    with open(os.path.join(src, "quant_model_description.json")) as f:
        desc = json.load(f)
    bad = {
        k: v
        for k, v in desc.items()
        if "." in k and not isinstance(v, str)
    }
    if bad:
        raise SystemExit(
            f"[FATAL] description json 张量条目出现非字符串值 {len(bad)} 个（示例: "
            f"{list(bad.items())[:2]}）—— 出现 dtype 字段，需同步后再运行 E7"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        default="outputs_npu/glm5/gptq/glm5-w4a8_moe_w8a8_dense",
        help="源产物目录（不修改）",
    )
    ap.add_argument("--dst", default=None, help="目标目录（默认 <src>_f32）")
    ap.add_argument("--dry-run", action="store_true", help="只列出将转换的张量，不写盘")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 dst")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst or (src + "_f32"))
    if not os.path.isdir(src):
        raise SystemExit(f"[FATAL] 源目录不存在: {src}")
    if os.path.exists(dst) and not args.overwrite:
        raise SystemExit(f"[FATAL] 目标已存在（--overwrite 覆盖）: {dst}")
    if os.path.abspath(dst) == os.path.abspath(src) or dst.startswith(src + os.sep):
        raise SystemExit("[FATAL] dst 必须在 src 之外")

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    targets = select_targets(weight_map)
    shards = sorted(set(weight_map.values()))
    shards_in_dir = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if shards != shards_in_dir:
        raise SystemExit(f"[FATAL] index 与目录 shard 清单不一致")

    print(f"src={src}")
    print(f"dst={dst}")
    print(f"targets: {len(targets)} tensors "
          f"({EXPECT_NORMS} norm + 1 lm_head), shards={len(shards)}")
    if args.dry_run:
        for name in sorted(targets):
            print("  cast -> F32:", name, "->", weight_map[name])
        return

    assert_description_no_dtype(src)

    os.makedirs(dst, exist_ok=True)
    # ---- copy non-shard files as-is ----
    for f in os.listdir(src):
        if f.endswith(".safetensors"):
            continue
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    # ---- rewrite shards ----
    src_headers: dict[str, dict] = {}
    cast_per_shard: dict[str, list[str]] = {}
    n_cast = 0
    for shard in shards:
        s_path = os.path.join(src, shard)
        d_path = os.path.join(dst, shard)
        header = read_header(s_path)
        src_headers[shard] = header
        meta = header.get("__metadata__")
        in_shard = [n for n in header if n != "__metadata__" and n in targets]
        if not in_shard:
            shutil.copy2(s_path, d_path)
            continue
        tensors = load_file(s_path)
        for name in in_shard:
            t = tensors[name]
            if t.dtype != torch.bfloat16:
                raise SystemExit(
                    f"[FATAL] {name} dtype={t.dtype}，预期 BF16（对齐 w4a8c8 只做 BF16→F32）"
                )
            tensors[name] = t.to(torch.float32)
        save_file(tensors, d_path, metadata=meta)
        del tensors
        cast_per_shard[shard] = in_shard
        n_cast += len(in_shard)
        print(f"  rewrote {shard}: {len(in_shard)} cast, "
              f"{len(header) - 1 - len(in_shard)} copied")

    # ---- post verify ----
    print("verify: rescanning dst headers ...")
    shard_dtypes_ok = True
    for shard in shards:
        h_src = src_headers[shard]
        h_dst = read_header(os.path.join(dst, shard))
        if set(h_src.keys()) != set(h_dst.keys()):
            raise SystemExit(f"[FATAL] {shard}: tensor set mismatch after rewrite")
        for name, info_s in h_src.items():
            if name == "__metadata__":
                meta_d = h_dst.get("__metadata__")
                if meta_d != info_s:
                    raise SystemExit(f"[FATAL] {shard}: metadata changed")
                continue
            info_d = h_dst[name]
            if name in targets:
                if info_d["dtype"] != "F32":
                    shard_dtypes_ok = False
                    raise SystemExit(
                        f"[FATAL] {shard}/{name}: dtype={info_d['dtype']}，预期 F32"
                    )
            else:
                if info_d["dtype"] != info_s["dtype"] or info_d["shape"] != info_s["shape"]:
                    raise SystemExit(
                        f"[FATAL] {shard}/{name}: 非 target 张量被改动 "
                        f"{info_s['dtype']}{info_s['shape']} -> {info_d['dtype']}{info_d['shape']}"
                    )
    # spot integrity check: sha256 of one byte-copied shard must match source
    copied = [s for s in shards if s not in cast_per_shard]
    if copied:
        spot = copied[0]
        h1, h2 = sha256_file(os.path.join(src, spot)), sha256_file(os.path.join(dst, spot))
        if h1 != h2:
            raise SystemExit(f"[FATAL] 字节复制 shard 校验失败: {spot}")
        print(f"verify: byte-copy spot check ok ({spot})")

    sz_src = sum(os.path.getsize(os.path.join(src, f)) for f in os.listdir(src))
    sz_dst = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    manifest = {
        "src": src,
        "dst": dst,
        "cast_tensors": sorted(targets.keys()),
        "cast_count": len(targets),
        "shards_rewritten": sorted(cast_per_shard.keys()),
        "shards_byte_copied": copied,
        "verify": {"dtypes_ok": shard_dtypes_ok, "casted": n_cast},
        "size_src_bytes": sz_src,
        "size_dst_bytes": sz_dst,
    }
    with open(os.path.join(dst, "postprocess_f32_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"done: cast {n_cast} tensors across {len(cast_per_shard)} shards; "
          f"byte-copied {len(copied)} shards")
    print(f"size: {sz_src / 1e9:.2f} GB -> {sz_dst / 1e9:.2f} GB "
          f"(+{(sz_dst - sz_src) / 1e9:.2f} GB，预期 ≈ lm_head 1.9 GB + norms 忽略不计)")


if __name__ == "__main__":
    main()
