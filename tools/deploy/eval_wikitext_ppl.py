#!/usr/bin/env python3

"""Quick wikitext-2 word-perplexity via the RUNNING vLLM completions API.

Mirrors lm_eval's `wikitext` task (EleutherAI/wikitext_document_level,
wikitext-2-raw-v1, output_type=loglikelihood_rolling) so the number is
comparable to the lm_eval reference. Runs entirely over HTTP - the vLLM
server keeps serving while this script measures it.

Usage:
    python3 eval_wikitext_ppl.py <pages.json> [--limit N] [--window 2048]

    pages.json: list[str] of wikitext page texts (validation split).
    --limit N: only evaluate the first N pages (quick smoke).
"""
from __future__ import annotations
# ===== Phase D 路径参数化（env 带默认回退）=====
import os
SHARED = os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = os.environ.get("YLT", SHARED + "/ylt")

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

API = "http://10.42.0.2:8000/v1/completions"
MODEL = "glm5-w4a16"
MODEL_DIR = f"{SHARED}/ylt/models/glm5-w4a16_npu_0807"
CONCURRENCY = 4


def wikitext_detokenizer(string: str) -> str:
    """Port of lm_eval lm_eval/tasks/wikitext/preprocess_wikitext.py."""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


@dataclass
class Result:
    sum_logprob: float
    words: int
    windows: int


def request_logprobs(prompt: str) -> float:
    """One window: sum of prompt-token logprobs (echo=True)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "echo": True,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    tlps = data["choices"][0]["logprobs"]["token_logprobs"]
    # layout: [null, prompt_lp_1, ..., prompt_lp_N, gen_lp]  (echo=True)
    prompt_lps = tlps[1:-1]
    assert prompt_lps and all(x is not None for x in prompt_lps), (
        f"unexpected logprobs layout: {tlps}"
    )
    return sum(prompt_lps)


def evaluate_page(page: str, tokenizer, window: int, executor) -> Result:
    detok = wikitext_detokenizer(page)
    ids = tokenizer(detok)["input_ids"]
    words = len(re.split(r"\s+", page))
    if not ids:
        return Result(0.0, words, 0)
    futures = []
    prompts = []
    for i in range(0, len(ids), window):
        chunk = ids[i : i + window]
        prompts.append(tokenizer.decode(chunk))
        futures.append(executor.submit(request_logprobs, prompts[-1]))
    total = sum(f.result() for f in futures)
    return Result(total, words, len(futures))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"[eval] loading tokenizer from {MODEL_DIR} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, use_fast=False
    )

    pages = json.load(open(args.pages_json))
    if args.limit:
        pages = pages[: args.limit]
    print(f"[eval] {len(pages)} pages, window={args.window}, concurrency={CONCURRENCY}", flush=True)

    t0 = time.time()
    all_sum = 0.0
    all_words = 0
    total_windows = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for idx, page in enumerate(pages):
            r = evaluate_page(page, tok, args.window, ex)
            all_sum += r.sum_logprob
            all_words += r.words
            total_windows += r.windows
            if (idx + 1) % 5 == 0 or idx == len(pages) - 1:
                print(
                    f"[eval] pages {idx+1}/{len(pages)} windows={total_windows} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

    word_ppl = math.exp(-all_sum / all_words)
    print("\n================= RESULTS =================")
    print(f"  pages   : {len(pages)}")
    print(f"  windows : {total_windows}")
    print(f"  words   : {all_words}")
    print(f"  sum(logp): {all_sum:.2f}")
    print(f"  word_perplexity: {word_ppl:.4f}")
    print(f"  elapsed : {time.time()-t0:.0f}s")
    print("===========================================")


if __name__ == "__main__":
    main()
