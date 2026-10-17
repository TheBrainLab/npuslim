#!/usr/bin/env python3
"""LiveCodeBench v6 (code_generation_lite, 1055 problems) generation probe.

Protocol (fixed): chat endpoint, system = official LCB system message,
user = official generic prompt template, enable_thinking=false, temperature=0,
max_tokens=2048. Saves {question_id, source_file, output, finish} per line.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LCB_DIR = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/lcb"
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

SYSTEM = ("You are an expert Python programmer. You will be given a question "
          "(problem specification) and you will generate a correct Python program that "
          "matches the specification and passes all tests.")
FMT_STARTER = ("You will use the following starter code to write the solution to the "
               "problem and enclose your code within delimiters.")
FMT_PLAIN = ("Read the inputs from stdin solve the problem and write the answer to stdout "
             "(do not directly test on the sample inputs). Enclose your code within delimiters "
             "as follows. Ensure that when the python program runs, it reads the inputs, runs "
             "the algorithm and writes output to STDOUT.")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def build_user(p: dict) -> str:
    s = f"### Question:\n{p['question_content']}\n\n"
    if p.get("starter_code"):
        s += f"### Format: {FMT_STARTER}\n```python\n{p['starter_code']}\n```\n\n"
    else:
        s += f"### Format: {FMT_PLAIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    s += "### Answer: (use the provided format with backticks)\n\n"
    return s


def post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    problems = []
    for f in FILES:
        for line in open(f"{LCB_DIR}/{f}"):
            d = json.loads(line)
            d["_file"] = f
            problems.append(d)
    print(f"loaded {len(problems)} problems", flush=True)

    def one(p):
        rec = {"question_id": p["question_id"], "source_file": p["_file"],
               "difficulty": p["difficulty"]}
        try:
            resp = post(args.api, {
                "model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_user(p)}]})
            ch = resp["choices"][0]
            rec.update({"output": ch["message"]["content"] or "",
                        "finish": ch.get("finish_reason")})
        except Exception as e:
            rec.update({"output": "", "error": repr(e)[:200]})
        return rec

    n = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for fut in as_completed(futs):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if n % 100 == 0:
                print(f"  [{n}/{len(problems)}]", flush=True)
    print(f"done: {n} generations -> {args.out}")


if __name__ == "__main__":
    main()
