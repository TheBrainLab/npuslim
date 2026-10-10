#!/usr/bin/env python3
"""Build the mixed-domain calibration JSONL WITH a gsm8k CoT domain added.

★ 背景（0908 §6c 诊断）：现有校准集 mixed_chat_code.jsonl 的 math 域是 modelslim
mix_calib（48 条选择题，short QA），**完全不含 gsm8k 式"分步算术推理 + 长 CoT 回答"**。
诊断①证明 gsm8k 缺口是**纯能力缺口（格式项≈0）**，且所有配方轴（seed/SQ/omni/AWQ）
都补齐不了 → 唯一未测过的轴 = **校准域"类型"**（历次实验只动过数据量/采样量，没动过类型）。

本脚本 = prepare_calib_data.py + gsm8k CoT 域（唯一差异）。
输出每条格式与 MixedDomainDataset 约定一致：{"domain": ..., "inputs_pretokenized"|"text"|"messages": ...}

用法: python tools/prepare_calib_data_cot.py [output_path] [n_c4] [n_gsm8k]
"""
import gzip
import json
import random
import sys
from pathlib import Path

SRC = Path("/data/yult/llm_inference/npuslim/dataset")
MS_LAB = Path("/data/yult/llm_inference/msmodelslim/lab_calib")
GSM8K_TRAIN = Path(
    "/data/yult/llm_inference/npuslim/workspace/hf_datasets_cache_4task/"
    "openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow"
)
DEFAULT_OUT = SRC / "calib" / "mixed_cot.jsonl"
DEFAULT_N_C4 = 128
DEFAULT_N_GSM8K = 64
C4_MIN_CHARS = 8000


def load_lines(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_c4_samples(n: int):
    gz = SRC / "c4" / "c4-train.00000.json.gz"
    pool = []
    if not gz.exists():
        return pool
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if len(text) >= C4_MIN_CHARS:
                pool.append(text)
            if len(pool) >= n * 4:
                break
    random.shuffle(pool)
    return pool[:n]


def load_gsm8k_cot(n: int):
    """gsm8k train split → 分步 CoT 文本（question + reasoning + #### answer）。

    用 train split 而非 test，避免与评测集重叠。
    序列化为 gsm8k 8-shot 的同一风格，使校准激活分布贴近评测时的长生成分布。
    """
    import pyarrow as pa
    if not GSM8K_TRAIN.exists():
        return []
    try:
        tbl = pa.ipc.open_file(str(GSM8K_TRAIN)).read_all()
    except Exception:
        tbl = pa.ipc.open_stream(str(GSM8K_TRAIN)).read_all()
    rows = tbl.to_pylist()
    random.shuffle(rows)
    out = []
    for r in rows[:n]:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or "####" not in a:
            continue
        # 与评测 few-shot 完全同风格（Question:/Answer: + CoT + #### N）
        text = f"Question: {q}\nAnswer: {a}"
        out.append(text)
    return out


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    n_c4 = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_C4
    n_gsm8k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_GSM8K

    random.seed(42)
    chat = load_lines(SRC / "sharegpt_gpt4_qwen" / "sharegpt_gpt4-qwen3_a22B_output.jsonl")
    code = load_lines(MS_LAB / "autocodebench.jsonl")
    math_ = load_lines(MS_LAB / "mix_calib.jsonl")
    web = load_c4_samples(n_c4)
    cot = load_gsm8k_cot(n_gsm8k)

    records = []
    for r in chat:
        if r.get("messages"):
            records.append({"domain": "chat", "messages": r["messages"]})
    for r in code:
        if r.get("inputs_pretokenized"):
            records.append({"domain": "code", "inputs_pretokenized": r["inputs_pretokenized"]})
    for r in math_:
        if r.get("inputs_pretokenized"):
            records.append({"domain": "math", "inputs_pretokenized": r["inputs_pretokenized"]})
    for text in web:
        records.append({"domain": "web", "text": text})
    for text in cot:
        records.append({"domain": "cot", "inputs_pretokenized": text})

    random.shuffle(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in records:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    print(f"written {len(records)} samples -> {out_path}")
    print(f"  domain mix: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
