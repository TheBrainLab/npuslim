#!/usr/bin/env python3
"""Smoke-test the running completions server: token-id prompt + echo + top-k logprobs.

Validates that (a) `prompt` accepts a list of token ids, (b) `echo=True` returns
one logprob per prompt position, (c) `--return-tokens-as-token-ids` makes the
`tokens` / `top_logprobs` keys integer token-id strings.
"""
import json
import sys
import urllib.request

api = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1/completions"
model = sys.argv[2] if len(sys.argv) > 2 else "glm52-w4a8-eval"

# "The capital of France is" - arbitrary small prompt as token ids (fallback text)
ids = [1, 200, 300, 400, 500]
payload = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
           "logprobs": 5, "echo": True}
req = urllib.request.Request(api, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.loads(r.read().decode())
lp = d["choices"][0]["logprobs"]
print("tokens:", lp["tokens"][:6])
print("token_logprobs:", lp["token_logprobs"][:6])
print("top_logprobs[1]:", lp["top_logprobs"][1])
print("n_prompt_lp:", len(lp["token_logprobs"]), "n_input:", len(ids))
