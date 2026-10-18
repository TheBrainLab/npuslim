#!/usr/bin/env python3
"""LiveCodeBench v6 scoring with the official testing_util (vendored copy).

pass@1: a problem passes iff ALL public+private test cases pass (official run_test,
timeout 6s). Code extraction: official last-two-fences rule; lenient fallback to the
raw output when no fences are present (documented; keeps style-band fairness).

Usage:
  python3 lcb_score.py --gen <generations.jsonl> --out <summary.json> [--workers 24]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import subprocess
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcb_testing_util import run_test  # noqa: E402  (vendored official)

LCB_DIR = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/lcb"
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def decode_private(s: str):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def load_problems() -> dict:
    probs = {}
    for f in FILES:
        if not os.path.exists(f"{LCB_DIR}/{f}"):
            print(f"  WARN: missing {f} - skipped", flush=True)
            continue
        for line in open(f"{LCB_DIR}/{f}"):
            d = json.loads(line)
            pub = json.loads(d["public_test_cases"])
            priv = decode_private(d["private_test_cases"])
            meta = json.loads(d.get("metadata") or "{}")
            probs[d["question_id"]] = {
                "sample": {"input_output": json.dumps({
                    "inputs": [t["input"] for t in pub + priv],
                    "outputs": [t["output"] for t in pub + priv],
                    "fn_name": meta.get("func_name"),
                })},
                "difficulty": d["difficulty"],
                "platform": d["platform"],
                "contest_date": d["contest_date"],
                "source_file": f,
                "n_tests": len(pub) + len(priv),
            }
    return probs


def extract_code(out: str):
    lines = out.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) >= 2:
        return "\n".join(lines[idx[-2] + 1: idx[-1]]), False
    return out.strip(), True


def grade(task):
    # [1017] isolate each problem in a subprocess: a candidate segfault (deep
    # recursion -> C-stack overflow) must not poison the pool (BrokenProcessPool).
    qid, sample, code = task
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_grade_one.py")],
            input=json.dumps({"sample": sample, "code": code}),
            capture_output=True, text=True, timeout=300)
        if p.returncode != 0:
            return qid, [-6], {"error": f"isolated child rc={p.returncode}"}
        out = json.loads(p.stdout.strip().splitlines()[-1])
        return qid, out["res"], out.get("meta", {})
    except subprocess.TimeoutExpired:
        return qid, [-6], {"error": "isolated child timeout"}
    except Exception as e:
        return qid, [-6], {"error": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    probs = load_problems()
    gens = [json.loads(l) for l in open(args.gen)]
    print(f"problems={len(probs)} generations={len(gens)}", flush=True)

    tasks, fallback = [], 0
    for g in gens:
        qid = g["question_id"]
        if qid not in probs or not g.get("output"):
            continue
        code, fb = extract_code(g["output"])
        fallback += int(fb)
        tasks.append((qid, probs[qid]["sample"], code))

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(grade, t) for t in tasks]
        for fut in as_completed(futs):
            qid, res, meta = fut.result()
            results[qid] = (res, meta)
            done += 1
            if done % 100 == 0:
                print(f"  graded [{done}/{len(tasks)}]", flush=True)

    agg = {"n": len(tasks), "passed": 0, "fallback_extract": fallback,
           "by_difficulty": {}, "by_platform": {}, "by_source_file": {}, "errors": 0,
           "trunc": 0}
    for g in gens:
        if g.get("finish") == "length":
            agg["trunc"] += 1
    detail = []
    for g in gens:
        qid = g["question_id"]
        if qid not in results:
            continue
        res, meta = results[qid]
        # official markers: True = case passed; -2 WA / -3 TLE / -4 RE / -5 runner error
        ok = bool(res) and all(x == True for x in res)  # noqa: E712
        agg["passed"] += int(ok)
        agg["errors"] += int(any(x in (-3, -4, -5, -6) for x in res if isinstance(x, int)))
        p = probs[qid]
        for key, bucket in (("difficulty", "by_difficulty"), ("platform", "by_platform"),
                            ("source_file", "by_source_file")):
            b = agg[bucket].setdefault(p[key], [0, 0])
            b[0] += int(ok)
            b[1] += 1
        detail.append({"question_id": qid, "passed": ok, "n_tests": p["n_tests"],
                       "difficulty": p["difficulty"], "platform": p["platform"],
                       "source_file": p["source_file"], "contest_date": p["contest_date"],
                       "res": res if len(res) < 40 else f"<{len(res)} cases>",
                       "finish": g.get("finish")})

    n = max(agg["n"], 1)
    print("=== LiveCodeBench v6 summary ===")
    print(f"  pass@1 = {agg['passed']}/{agg['n']} = {agg['passed']/n:.3f} "
          f"(extract-fallback {agg['fallback_extract']}, errors {agg['errors']}, trunc {agg['trunc']})")
    for bucket in ("by_difficulty", "by_platform", "by_source_file"):
        print(f"  {bucket}:")
        for k, (ok, tot) in sorted(agg[bucket].items()):
            print(f"    {k}: {ok}/{tot} = {ok/max(tot,1):.3f}")
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=1)
    with open(args.out.replace(".json", "_detail.jsonl"), "w") as f:
        for d in detail:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
