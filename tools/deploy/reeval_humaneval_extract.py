#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reeval_humaneval_extract.py — humaneval 标准 extract_code 口径重评（可跑已落盘 samples）

标准做法（OpenAI HumanEval / 主流 harness）：先从 completion 提取 ```python ... ``` 代码块，
再送 code_eval。本脚本与 reeval_humaneval_trunc.py 同接口，便于与截断口径交叉验证：
  raw（原始 harness） / trunc（幻觉标记截断） / extract-first / extract-last
用法: HF_ALLOW_CODE_EVAL=1 python3 reeval_humaneval_extract.py [tag=path ...]
"""
import importlib.util
import re
import sys

RH = "/llm_infer_workspace/ylt/npuslim/workspace/glm5.2_deploy/reeval_humaneval_trunc.py"
spec = importlib.util.spec_from_file_location("rh", RH)
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)
check_correctness = rh.check_correctness

FENCE = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)[ \t]*\n?(.*?)```", re.DOTALL)


def _base_program(comp: str, prompt: str) -> str:
    """返回"待提取"的完整程序。

    本评测的 filtered_resps 是**完整全文**（模型回显 prompt + 续写 + METADATA 尾），
    因此不能剥离 prompt（否则会删掉 `from typing import List` 导致 NameError）。
    仅当 comp 明显不含 prompt（例如只输出续写）时才前置 prompt。
    """
    if prompt:
        head = prompt.strip()[:30]
        if head and head not in comp[:400]:
            return prompt + comp
    return comp


def extract_first(comp: str, prompt: str = "") -> str:
    ms = FENCE.findall(_base_program(comp, prompt))
    return ms[0] if ms else rh.truncate(comp)


def extract_last(comp: str, prompt: str = "") -> str:
    ms = FENCE.findall(_base_program(comp, prompt))
    return ms[-1] if ms else rh.truncate(comp)


O = "/llm_infer_workspace/ylt/glm5.2_deploy/outputs"
DEFAULT = [
    ("bf16", f"{O}/glm5_bf16_4task_hard/qdpv68r5/samples_humaneval_2026-09-03T23-39-35.499266.jsonl"),
    ("w4a8c8", f"{O}/glm5_w4a8c8_4task_hard/qdpv68r5/samples_humaneval_2026-09-04T05-36-06.906183.jsonl"),
    ("w4a8_self", f"{O}/glm5_w4a8_4task_hard/qdpv68r5/samples_humaneval_2026-09-03T09-51-50.955236.jsonl"),
    ("p1base_l0", f"{O}/glm5_w4a8_p1base_l0_4task_hard/qdpv68r5/samples_humaneval_2026-09-11T11-25-36.599405.jsonl"),
    ("l15fix2", f"{O}/glm5_l15fix2_fast/qdpv68r5/samples_humaneval_2026-09-12T11-19-11.165954.jsonl"),
    ("path2fix2", f"{O}/glm5_l15p2fix2_fast/qdpv68r5/samples_humaneval_2026-09-12T11-19-57.131550.jsonl"),
    ("l15fix_broken", f"{O}/glm5_w4a8_l15fix_4task_hard/qdpv68r5/samples_humaneval_2026-09-12T06-23-45.491729.jsonl"),
]


def run(path, tag):
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = []
    for line in open(path):
        d = _json.loads(line)
        r = d.get("filtered_resps") or d.get("resps")
        comp = r[0][0] if r and isinstance(r[0], list) else (r[0] if r else "")
        rows.append((d["doc"]["task_id"], comp, d["doc"]["test"],
                     d["doc"].get("prompt", ""), bool(d.get("pass@1", 0))))
    n = len(rows)
    raw = sum(x[4] for x in rows)
    res = {}
    for mode, fn in (("first", extract_first), ("last", extract_last),
                     ("trunc", lambda c, p="": rh.truncate(c))):
        passed = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {}
            for tid, comp, test, prompt, _ in rows:
                prog = fn(comp, prompt) + "\n" + test
                futs[ex.submit(check_correctness, prog, 3.0, tid, 0)] = tid
            for f in as_completed(futs):
                if f.result()["passed"]:
                    passed += 1
        res[mode] = passed / n
    print(f"{tag:14s} n={n} raw={raw/n:.4f}  extract_first={res['first']:.4f}  "
          f"extract_last={res['last']:.4f}  trunc={res['trunc']:.4f}")
    return res


def main():
    pairs = []
    for a in sys.argv[1:]:
        t, p = a.split("=", 1)
        pairs.append((t, p))
    if not pairs:
        pairs = DEFAULT
    out = {}
    for tag, path in pairs:
        out[tag] = run(path, tag)
    print("\n===== 口径交叉验证（排序一致性）=====")
    for mode in ("first", "last", "trunc"):
        order = sorted(out, key=lambda t: -out[t][mode])
        print(f"  {mode:6s}: " + " > ".join(f"{t}({out[t][mode]:.4f})" for t in order))


if __name__ == "__main__":
    main()
