#!/usr/bin/env python3
"""RULER-hard v3 (1017): var_track chain-length calibration x dual mode.

Context: v2 (40-step chain, non-think) floored the model (p2clip 0/12).
v3 calibrates: chain lengths {12, 24, 40} x 3 cases x modes {non-think, think}
= 18 generations per product, in ONE 64K round.

Protocol:
  * context length 60000 tokens (60K class; leaves room for think output under maxlen 69632)
  * non-think: enable_thinking=false, max_tokens=64
  * think    : enable_thinking=true,  max_tokens=4096 (equal budget for all products)
  * temperature=0
Scoring: gold (small int) as a boundary-anchored number in the FINAL answer text;
inline  thinking...</think> blocks are stripped; if the API returns a separate
reasoning_content field, only the final content is searched.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request

WIKITEXT = "/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd_data/wikitext_train.txt"
CHARS_PER_TOKEN = 4.0
LENGTHS = [12, 24, 40]
PER_LENGTH = 3

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build(hay: str, n_chars: int, inserts: list[tuple[float, str]], query: str) -> str:
    body = hay[:n_chars]
    out, prev = [], 0
    for frac, snip in sorted(inserts, key=lambda x: x[0]):
        pos = max(prev + 1, min(int(frac * len(body)), len(body)))
        out.append(body[prev:pos])
        out.append("\n" + snip + "\n")
        prev = pos
    out.append(body[prev:])
    return "".join(out) + f"\n\nQuestion: {query}\nAnswer:"


def num_in(ans: str, gold: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(gold)}(?!\d)", ans) is not None


THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S)


def final_text(content: str, reasoning: str | None) -> str:
    if reasoning is not None:
        return content  # thinking separated by the server -> final answer only
    return THINK_RE.sub("", content)


def make_chain(rng: random.Random, n_steps: int):
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    cur_name = rng.choice(names)
    cur_val = rng.randint(1, 20)
    steps = [f"{cur_name} = {cur_val}."]
    for _ in range(n_steps):
        nxt = rng.choice([n for n in names if n != cur_name])
        op = rng.choice(["+", "-"])
        k = rng.randint(2, 9)
        steps.append(f"{nxt} = {cur_name} {op} {k}.")
        cur_val = cur_val + k if op == "+" else cur_val - k
        cur_name = nxt
    return steps, cur_name, cur_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3031)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hay = load_haystack()
    n_chars = min(int(args.length * CHARS_PER_TOKEN), len(hay))
    cases = []
    for L in LENGTHS:
        for _ in range(PER_LENGTH):
            steps, last_name, gold = make_chain(rng, L)
            inserts = [(0.05 + 0.9 * (j / len(steps)), s) for j, s in enumerate(steps)]
            q = (f"In the provided text there is a sequence of assignments; "
                 f"what is the FINAL value assigned to the variable {last_name}?")
            cases.append({"length": L, "gold": str(gold), "prompt": build(hay, n_chars, inserts, q)})

    def one(case, mode):
        think = mode == "think"
        out = {"task": "var_track_chain", "length": case["length"], "mode": mode, "gold": case["gold"]}
        try:
            resp = post(args.api, {
                "model": args.model, "temperature": 0,
                "max_tokens": 4096 if think else 64,
                "chat_template_kwargs": {"enable_thinking": think},
                "messages": [{"role": "user", "content": case["prompt"]}]})
            ch = resp["choices"][0]
            msg = ch["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content")
            fin = final_text(content, reasoning)
            out.update({"correct": int(num_in(fin, case["gold"])),
                        "in_full": int(num_in(content + (reasoning or ""), case["gold"])),
                        "finish": ch.get("finish_reason"),
                        "n_out_tok": ch.get("usage", {}).get("completion_tokens"),
                        "answer": fin.strip()[-800:]})
        except Exception as e:
            out.update({"correct": 0, "error": repr(e)[:200]})
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = [(c, m) for c in cases for m in ("nonthink", "think")]
    summary = {}
    n_done = 0
    with open(args.out, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, c, m) for c, m in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            key = f"L{r['length']}_{r['mode']}"
            s = summary.setdefault(key, {"n": 0, "correct": 0, "err": 0, "trunc": 0})
            s["n"] += 1
            s["correct"] += r.get("correct", 0)
            s["err"] += int(bool(r.get("error")))
            s["trunc"] += int(r.get("finish") == "length")
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            print(f"  [{n_done}/{len(tasks)}] L{r['length']} {r['mode']} correct={r.get('correct')} "
                  f"gold={r['gold']} fin={r.get('finish')}", flush=True)

    print(f"=== summary (L={args.length}, seed={args.seed}) ===")
    for k, s in sorted(summary.items()):
        print(f"  {k}: {s['correct']}/{s['n']} (trunc={s['trunc']}, err={s['err']})")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def load_haystack() -> str:
    return open(WIKITEXT).read()


if __name__ == "__main__":
    main()
