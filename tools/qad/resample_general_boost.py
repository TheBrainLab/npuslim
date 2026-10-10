#!/usr/bin/env python
"""resample_general_boost.py — 通识数据重采样（mmlu 缓解，数据侧零代码零 precompute）。

qad_train_v1.jsonl 的 E-general 条目 ×3 复制（42.6% → 69.4%），其余 ×1，
shuffle 后写 qad_train_v2.jsonl。复制条目的 sample_key 不变 → teacher cache
命中同一行（零 precompute）。训练经 QAD_TRAIN_TABLE env 使用 v2。
"""
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
NPUSLIM_SRC = _pd_os.environ.get("NPUSLIM_SRC", YLT + "/npuslim/src")
QAD_ROOT = _pd_os.environ.get("QAD_ROOT", YLT + "/glm52-qad")
# =========================================================
import json, random, collections

SRC = f"{SHARED_HOST}/ylt/glm52-qad/qad_data/qad_train_v1.jsonl"
DST = f"{SHARED_HOST}/ylt/glm52-qad/qad_data/qad_train_v2_general3x.jsonl"
GENERAL_PREFIX = "E-general-nemotron-pt-v1.jsonl"
BOOST = 3

rows = []
for line in open(SRC, encoding="utf-8"):
    if line.strip():
        rows.append(line)

out, cnt = [], collections.Counter()
for line in rows:
    s = json.loads(line)
    sk = s.get("sample_key", "")
    reps = BOOST if sk.startswith(GENERAL_PREFIX) else 1
    cnt[reps] += 1
    for _ in range(reps):
        out.append(line)

random.Random(42).shuffle(out)
with open(DST, "w", encoding="utf-8") as f:
    f.writelines(out)

total = len(out)
gen = sum(1 for line in out if json.loads(line).get("sample_key", "").startswith(GENERAL_PREFIX))
print(f"src={len(rows)} out={total} general_entries={gen} "
      f"general_pct={gen*100/total:.1f}% -> {DST}", flush=True)
