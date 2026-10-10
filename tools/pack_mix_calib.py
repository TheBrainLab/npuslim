#!/usr/bin/env python3

# ===== Phase D 路径参数化（env 带默认回退）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "{SHARED}")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult{SHARED}")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
"""pack_mix_calib.py — 把 modelslim mix_calib.jsonl（48 条短样本）打包成 >1024 token 的 chunk。

目的：npuslim GPTQ 校准要求"等长 1024 窗口"，短于窗口的样本会被跳过
（MixedDomainDataset: `len <= max_length -> continue`）。把 48 条拼接后按
~1150 token 切片，使每个 chunk > 1024，从而全部内容可通过窗口进入校准。

输出：dataset/calib/modelslim_mix_calib_packed.jsonl（{"domain":"math","text":...}）
用法：python3 tools/pack_mix_calib.py [chunk_tokens]
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SRC = Path("{SHARED}/ylt/npuslim/dataset/calib/modelslim_mix_calib.jsonl")
TOK = "{SHARED}/models/GLM-5.2-bf16/snapshots/master"
OUT = Path("{SHARED}/ylt/npuslim/dataset/calib/modelslim_mix_calib_packed.jsonl")
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 1150


def main() -> int:
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    sep = [tok.eos_token_id] if tok.eos_token_id is not None else []
    all_ids = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line).get("inputs_pretokenized", "")
        if t:
            all_ids.extend(tok(t, add_special_tokens=False).input_ids)
            all_ids.extend(sep)
    chunks = [all_ids[i:i + CHUNK] for i in range(0, len(all_ids), CHUNK)]
    chunks = [c for c in chunks if len(c) > 1024]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"domain": "math", "text": tok.decode(c)}, ensure_ascii=False) + "\n")
    print(f"total_tokens={len(all_ids)}  chunks={len(chunks)} (each >1024) -> {OUT}")
    print(f"chunk token lens: min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
