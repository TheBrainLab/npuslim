#!/usr/bin/env python3
"""#3 P2b: BFCL v4 decision-axis probe (irrelevance / refusal-to-call + name-level relevance).

Protocol (fixed, official v4 default prompt):
  system = persona+task + tool_call_format(python) + multiturn + available_tools(JSON)
  (= DEFAULT_SYSTEM_PROMPT_FORMAT "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic")
  user   = the case question
  metric:
    * irrelevance / live_irrelevance : CORRECT = model emits NO function call  (refusal-to-call rate)
    * simple_python                  : CORRECT = model calls the RIGHT function (name-level check)

Runs on the control host (urllib only); talks to the vLLM server over the node IP.
Writes per-case JSONL (for paired analysis) + a summary JSON.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import time
import urllib.request

PERSONA = "You are an expert in composing functions."
TASK = ("You are given a question and a set of possible functions. Based on the question, "
        "you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. If the given question lacks "
        "the parameters required by the function, also point it out.")
TOOL_CALL_FORMAT = ("You should only return the function calls in your response.\n\n"
                    "If you decide to invoke any of the function(s), you MUST put it in the format of "
                    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]."
                    "  You SHOULD NOT include any other text in the response.")
MULTITURN = ("At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
             "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
             "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.")


def build_system(funcs: list[dict]) -> str:
    tools = "Here is a list of functions in JSON format that you can invoke.\n" + \
            json.dumps(funcs, indent=4, ensure_ascii=False) + "\n"
    return f"{PERSONA}{TASK}\n\n{TOOL_CALL_FORMAT}\n\n{MULTITURN}\n\n{tools}"


def detect_call(text: str, func_names: list[str]) -> bool:
    """Did the model emit a function call? (official python-format first, regex fallback)."""
    t = text.strip()
    # strip markdown fences
    t = re.sub(r"^```[a-zA-Z0-9_+\-]*\s*|\s*```$", "", t, flags=re.DOTALL).strip()
    # 1) official: a python list literal of calls
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end > start:
        frag = t[start:end + 1]
        for cand in (frag, frag.replace("'", '"')):
            try:
                obj = ast.literal_eval(cand)
            except Exception:
                continue
            items = obj if isinstance(obj, (list, tuple)) else [obj]
            for it in items:
                name = None
                if isinstance(it, dict):
                    name = it.get("name") or it.get("function")
                elif isinstance(it, (list, tuple)) and it and isinstance(it[0], str):
                    name = it[0]
                if name in func_names:
                    return True
    # 2) fallback: name( ... ) patterns
    for name in func_names:
        if re.search(re.escape(name) + r"\s*\(", t):
            return True
    return False


# Local API calls must bypass any host proxy (坑51: 10.1.0.x 会被死代理劫持 -> 502)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="http://<node>:<port>/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cases = []  # (category, case)
    for cat, fn in [("irrelevance", "BFCL_v4_irrelevance.json"),
                    ("live_irrelevance", "BFCL_v4_live_irrelevance.json"),
                    ("simple_python", "BFCL_v4_simple_python.json")]:
        path = f"{args.data_dir}/{fn}"
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if args.limit:
            rows = rows[:args.limit]
        cases += [(cat, r) for r in rows]

    out_path = args.out
    summary = {c: {"n": 0, "correct": 0} for c, _ in cases}

    def one(item):
        cat, r = item
        funcs = r["function"]
        names = [f.get("name") for f in funcs]
        q = r["question"]
        user = ""
        for msg in q[0]:
            if msg.get("role") == "user":
                user = msg["content"]
        payload = {"model": args.model, "temperature": 0, "max_tokens": 256,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "system", "content": build_system(funcs)},
                                {"role": "user", "content": user}]}
        rec = {"id": r["id"], "category": cat, "names": names}
        try:
            resp = post(args.api, payload)
            content = resp["choices"][0]["message"]["content"]
            called = detect_call(content, names)
            correct = (not called) if cat.endswith("irrelevance") else called
            rec.update({"called": called, "correct": correct, "content": content[:1500]})
        except Exception as e:
            rec.update({"error": repr(e)[:200], "correct": False})
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_done = 0
    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c) for c in cases]
        for fut in as_completed(futs):
            rec = fut.result()
            summary[rec["category"]]["n"] += 1
            summary[rec["category"]]["correct"] += int(bool(rec.get("correct")))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [{n_done}/{len(cases)}]", flush=True)

    print("=== summary ===")
    for c, s in summary.items():
        if s["n"]:
            print(f"  {c}: {s['correct']}/{s['n']} = {s['correct']/s['n']:.4f}")
    with open(out_path.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
