#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""humaneval 截断重评：基于已落盘 samples，在幻觉标记处截断 completion 后重跑 code_eval。

纯离线（本地 CPU 执行 Python 测试），不需要 vLLM 推理服务。
用法：
  HF_ALLOW_CODE_EVAL=1 python3 reeval_humaneval_trunc.py \
      <w4a8c8_samples.jsonl> <w4a8_samples.jsonl> <bf16_samples.jsonl>
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# check_correctness 在同目录 execute.py（evaluate.metrics 不可用，直接从文件加载）
_EXECUTE_PATH = "/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8/execute.py"
_spec = importlib.util.spec_from_file_location("humaneval_execute", _EXECUTE_PATH)
_execute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_execute)
check_correctness = _execute.check_correctness

HALLUC_MARKERS = ("```", "METADATA", "</file>", "<file ", "<|file_sep|>", "<file_sep>",
                  "<file>", "<file_start", "# === ", "def test_")


def truncate(comp):
    best = None
    for m in HALLUC_MARKERS:
        i = comp.find(m)
        if i != -1 and (best is None or i < best[1]):
            best = (m, i)
    if best is None:
        return comp
    return comp[: best[1]].rstrip()


def load(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        resps = d.get("filtered_resps") or d.get("resps")
        comp = ""
        if resps and resps[0]:
            comp = resps[0][0] if isinstance(resps[0], list) else resps[0]
        out.append((d["doc"]["task_id"], comp, d["doc"]["test"], bool(d.get("pass@1", 0))))
    return out


def reeval(path, tag):
    samples = load(path)
    n = len(samples)
    raw_pass = sum(s[3] for s in samples)
    fut_map = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for tid, comp, test, _ in samples:
            tcomp = truncate(comp)
            prog = tcomp + "\n" + test
            fut_map[ex.submit(check_correctness, prog, 3.0, tid, 0)] = (tid, comp, tcomp)
        passed = 0
        truncated_recovered = 0  # 原始失败但截断后通过
        for f in as_completed(fut_map):
            tid, comp, tcomp = fut_map[f]
            r = f.result()
            ok = r["passed"]
            if ok:
                passed += 1
                # 原始是否失败？
                orig = [s[3] for s in samples if s[0] == tid]
                if orig and not orig[0]:
                    truncated_recovered += 1
    print(f"{tag}: n={n}  raw pass@1={raw_pass} ({raw_pass/n:.4f})  "
          f"TRUNCATED pass@1={passed} ({passed/n:.4f})  "
          f"截断恢复={truncated_recovered}")
    return raw_pass / n, passed / n


def main():
    paths = {"w4a8c8": sys.argv[1], "w4a8": sys.argv[2], "bf16": sys.argv[3]}
    res = {}
    for tag in ("bf16", "w4a8", "w4a8c8"):
        res[tag] = reeval(paths[tag], tag)
    print("\n===== 截断后对比（剥离口径污染后的真实量化差距） =====")
    for tag in ("bf16", "w4a8", "w4a8c8"):
        raw, trunc = res[tag]
        print(f"{tag:7s} raw={raw:.4f}  trunc={trunc:.4f}  Δtrunc(截断提升)={trunc-raw:+.4f}")
    print(f"w4a8c8 vs bf16:  raw Δ={(res['w4a8c8'][0]-res['bf16'][0])*100:+.2f}pp  "
          f"trunc Δ={(res['w4a8c8'][1]-res['bf16'][1])*100:+.2f}pp")
    print(f"w4a8c8 vs w4a8:  raw Δ={(res['w4a8c8'][0]-res['w4a8'][0])*100:+.2f}pp  "
          f"trunc Δ={(res['w4a8c8'][1]-res['w4a8'][1])*100:+.2f}pp")


if __name__ == "__main__":
    main()
