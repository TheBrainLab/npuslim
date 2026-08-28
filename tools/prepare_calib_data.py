#!/usr/bin/env python3
"""Build the mixed-domain calibration JSONL (chat + code + math + C4 web) for npuslim.

Sources:
  - chat : npuslim dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
           (255 samples, {"messages": [...]})
  - code : msmodelslim lab_calib/autocodebench.jsonl (43, {"inputs_pretokenized": ...})
  - math : msmodelslim lab_calib/mix_calib.jsonl (48, {"inputs_pretokenized": ...})
  - web  : npuslim dataset/c4/c4-train.00000.json.gz (long general-domain documents;
           guarantees enough samples survive the max_seq_length window filter --
           chat/code samples are often shorter than 1024 tokens and get skipped)

Output lines keep the source format untouched (MixedDomainDataset auto-detects
`messages` vs `text` vs `inputs_pretokenized`), tagged with a "domain" field:
  {"domain": "chat", "messages": [...]}
  {"domain": "code", "inputs_pretokenized": "..."}
  {"domain": "web",  "text": "..."}

Usage: python tools/prepare_calib_data.py [output_path] [n_c4]
"""
import gzip
import json
import random
import sys
from pathlib import Path

SRC = Path("/data/yult/llm_inference/npuslim/dataset")
MS_LAB = Path("/data/yult/llm_inference/msmodelslim/lab_calib")
DEFAULT_OUT = SRC / "calib" / "mixed_chat_code.jsonl"
DEFAULT_N_C4 = 128
# Only keep C4 docs comfortably longer than the 1024-token window (~4 chars/token)
C4_MIN_CHARS = 8000


def load_lines(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_c4_samples(n: int):
    """Sample n long C4 documents from the local json.gz shard."""
    with gzip.open(SRC / "c4" / "c4-train.00000.json.gz", "rt", encoding="utf-8") as f:
        pool = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if len(text) >= C4_MIN_CHARS:
                pool.append(text)
            if len(pool) >= n * 4:  # oversample then downsample
                break
    random.shuffle(pool)
    return pool[:n]


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    n_c4 = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_C4

    chat = load_lines(SRC / "sharegpt_gpt4_qwen" / "sharegpt_gpt4-qwen3_a22B_output.jsonl")
    code = load_lines(MS_LAB / "autocodebench.jsonl")
    math_ = load_lines(MS_LAB / "mix_calib.jsonl")
    web = load_c4_samples(n_c4)

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

    random.seed(42)
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
