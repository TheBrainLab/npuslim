#!/usr/bin/env python3

# ===== Phase D 路径参数化（env 带默认回退）=====
import os
SHARED = os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = os.environ.get("YLT", SHARED + "/ylt")
# -*- coding: utf-8 -*-
"""humaneval 公平性排查：三版本（BF16/W4A8/w4a8c8）samples 对照分析。

复用 §14.5 方法：code_eval 直接把 candidate + test_case 拼成 test_program 运行（无 _extract_code），
模型 completion 中的续写幻觉（``` 、METADATA、</file>、<file 等）会导致 SyntaxError。

本脚本做静态对照：
  1. 逐样本 pass@1 对比（找 w4a8c8 失败但其他版本通过的样本）
  2. 各版本 completion 中幻觉标记占比
  3. 用 ast.parse(test_program=candidate+test) 静态判定语法错误（§14.5 显示 SyntaxError 为主因）
  4. 截断实验：在幻觉标记处截断后语法错误是否消失（可恢复性估计）
"""
import ast
import json
import sys
from collections import Counter, defaultdict

HALLUC_MARKERS = ("```", "METADATA", "</file>", "<file ", "<|file_sep|>", "<file_sep>",
                  "<file>", "<file_start", "<file_start>", "# === ", "def test_")

def load_samples(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        resps = d.get("filtered_resps") or d.get("resps")
        comp = ""
        if resps and resps[0]:
            comp = resps[0][0] if isinstance(resps[0], list) else resps[0]
        out.append({
            "task_id": d["doc"]["task_id"],
            "comp": comp,
            "test": d["doc"]["test"],
            "pass": bool(d.get("pass@1", 0)),
        })
    return out

def has_markers(comp):
    return [m for m in HALLUC_MARKERS if m in comp]

def truncate(comp):
    """在首个幻觉标记处截断。"""
    best = None
    for m in HALLUC_MARKERS:
        i = comp.find(m)
        if i != -1 and (best is None or i < best[1]):
            best = (m, i)
    if best is None:
        return comp, None
    return comp[:best[1]].rstrip(), best[0]

def syntax_ok(program):
    try:
        ast.parse(program)
        return True
    except SyntaxError:
        return False

def analyze(path):
    samples = load_samples(path)
    n = len(samples)
    n_pass = sum(s["pass"] for s in samples)
    mark_stats = Counter()
    fail_with_mark = 0
    fail_syntax = 0
    fail_syntax_recover = 0
    fail_syntax_still_bad = 0
    recoverable_syntax = 0
    # 全样本语法检查（candidate+test）
    raw_bad = 0
    for s in samples:
        marks = has_markers(s["comp"])
        if marks:
            mark_stats[marks[0]] += 1
        prog = s["comp"] + "\n" + s["test"]
        if not s["pass"]:
            if marks:
                fail_with_mark += 1
            if not syntax_ok(prog):
                fail_syntax += 1
                t, m = truncate(s["comp"])
                if syntax_ok(t + "\n" + s["test"]):
                    fail_syntax_recover += 1
                else:
                    fail_syntax_still_bad += 1
            else:
                recoverable_syntax += 0
        if not syntax_ok(prog):
            raw_bad += 1
    return {
        "n": n, "pass": n_pass, "rate": n_pass / n,
        "mark_stats": dict(mark_stats),
        "fail_with_mark": fail_with_mark,
        "fail_syntax": fail_syntax,
        "fail_syntax_recover": fail_syntax_recover,
        "fail_syntax_still_bad": fail_syntax_still_bad,
        "raw_syntax_bad": raw_bad,
        "samples": samples,
    }


def main():
    paths = {
        "w4a8c8": sys.argv[1],
        "w4a8": sys.argv[2],
        "bf16": sys.argv[3],
    }
    res = {k: analyze(v) for k, v in paths.items()}

    print("================ 各版本 humaneval 对照 ================")
    for k in ("bf16", "w4a8", "w4a8c8"):
        r = res[k]
        print(f"{k:7s} n={r['n']}  pass={r['pass']}  rate={r['rate']:.4f}  "
              f"幻觉标记样本={r['fail_with_mark']}  语法错={r['fail_syntax']}  "
              f"截断可恢复={r['fail_syntax_recover']}  截断仍错={r['fail_syntax_still_bad']}  "
              f"markers={r['mark_stats']}")

    # 逐样本差异：w4a8c8 失败但 bf16/w4a8 通过的样本
    by_id = {k: {s["task_id"]: s for s in res[k]["samples"]} for k in res}
    diff_ids = [tid for tid in by_id["w4a8c8"]
                if not by_id["w4a8c8"][tid]["pass"] and (by_id["bf16"][tid]["pass"] or by_id["w4a8"][tid]["pass"])]
    print(f"\n================ w4a8c8 失败但 BF16/W4A8 通过的样本: {len(diff_ids)} ================")
    for tid in diff_ids:
        c8 = by_id["w4a8c8"][tid]["comp"]
        marks = has_markers(c8)
        # 截断后语法
        t, m = truncate(c8)
        prog = c8 + "\n" + by_id["w4a8c8"][tid]["test"]
        tprog = t + "\n" + by_id["w4a8c8"][tid]["test"]
        print(f"  {tid}: bf16={'Y' if by_id['bf16'][tid]['pass'] else 'N'} "
              f"w4a8={'Y' if by_id['w4a8'][tid]['pass'] else 'N'} "
              f"c8=N  markers={marks}  raw_syntax={'BAD' if not syntax_ok(prog) else 'OK'} "
              f"trunc_syntax={'OK' if syntax_ok(tprog) else 'BAD'}")
        # 打印 completion 尾部（幻觉内容）前 120 字符
        if m is not None and t:
            tail = c8[t.rfind(m)-0:][:0]
        # 展示 candidate 的最后 160 字符，便于观察幻觉内容
        print("      ...tail:", repr(c8[-160:]))

    # 各版本失败样本中含幻觉标记的分布（全部失败样本，不只 diff）
    print("\n================ 各版本失败样本的幻觉标记占比 ================")
    for k in ("bf16", "w4a8", "w4a8c8"):
        r = res[k]
        fails = [s for s in r["samples"] if not s["pass"]]
        fm = sum(1 for s in fails if has_markers(s["comp"]))
        print(f"  {k}: 失败 {len(fails)} 个，含幻觉标记 {fm} 个 ({fm/max(len(fails),1):.2%})")


if __name__ == "__main__":
    main()
