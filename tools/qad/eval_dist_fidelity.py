#!/usr/bin/env python3
"""Distribution fidelity vs the offline BF16 teacher cache.

L0 instruments (see npuslim-docs/GLM-5.2-量化评测落地方案1015.md §2 L0):
  #2 decision-margin displacement  m' ≈ c·m + b
  #4 domain KL(teacher || student)   (top-100 + uniform-tail proxy)
  #5 top-1 / top-5 consistency
  #3 exact-token NLL gap  (a component of forward KL; PPL ratio = exp(gap))

Reference = the GLM-5.2 BF16 teacher top-K log-prob cache
(`glm52-qad/precompute/logits_cache`, top_k=100, log_softmax, vocab 154880).
Student   = the running W4A8 vLLM completions server.

Alignment: the cache row i holds P(.|enc[0..i]) and col-0 = token enc[i+1]
(= actual_ids[i]). We feed the student `[bos] + actual_ids`; with a correct bos
the student prompt position (i+1) predicts exactly enc[i+1] under context
enc[0..i] -> student position i+1 aligns with teacher row i.

Requires the server started with `--return-tokens-as-token-ids`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np


def req_logprobs(api, model, ids, topk, timeout):
    payload = {"model": model, "prompt": ids, "max_tokens": 1,
               "temperature": 0, "logprobs": topk, "echo": True}
    r = urllib.request.Request(api, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        d = json.loads(resp.read().decode())
    lp = d["choices"][0]["logprobs"]
    return lp["token_logprobs"], lp["top_logprobs"]


def kl_topk(pt_ids, pt_lps, ps_map, vocab, K):
    """KL(pT || pS) over teacher top-K (+uniform tail) vs student top-K (+uniform tail)."""
    pt = np.exp(np.asarray(pt_lps, dtype=np.float64))
    mass_t = pt.sum()
    tail_t = max(0.0, 1.0 - mass_t) / max(1, vocab - len(pt_ids))
    mass_s = sum(ps_map.values())
    tail_s = max(0.0, 1.0 - mass_s) / max(1, vocab - len(ps_map))
    kl = 0.0
    for tid, p in zip(pt_ids, pt):
        q = ps_map.get(int(tid))
        q = q if q is not None else tail_s
        if p > 0:
            kl += p * (math.log(p) - math.log(max(q, 1e-12)))
    return kl


def load_refs(index_path, min_rows=128):
    by_shard = defaultdict(list)
    n = 0
    keep = 0
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if int(r.get("logit_rows", 0)) >= min_rows:
                by_shard[r["shard"]].append(r)
                keep += 1
    return by_shard, n, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", default="glm52-w4a8-eval")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--per-shard", type=int, default=40)
    ap.add_argument("--min-rows", type=int, default=128, help="min logit_rows (sample length) to include")
    ap.add_argument("--shard-stride", type=int, default=0, help="pick shards evenly spaced (0=first n_shards)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    bos = tok.bos_token_id
    prefix = [int(bos)] if bos is not None else []
    print(f"[fid] tokenizer loaded; bos={bos} prefix_len={len(prefix)}", flush=True)

    by_shard, n_total, n_keep = load_refs(f"{args.cache}/index.jsonl", args.min_rows)
    keys = sorted(by_shard.keys())
    if args.shard_stride and len(keys) > args.n_shards:
        idx = [int(round(i * (len(keys) - 1) / (args.n_shards - 1))) for i in range(args.n_shards)] if args.n_shards > 1 else [0]
        shards = [keys[i] for i in sorted(set(idx))]
    else:
        shards = keys[: args.n_shards]
    print(f"[fid] cache {n_total} samples, {n_keep} with >= {args.min_rows} rows; using {shards}", flush=True)

    agg = defaultdict(float)
    n_pos = 0
    mt_list, ms_list = [], []
    per_sample = []
    t_start = time.time()
    for shard in shards:
        path = f"{args.cache}/{shard}.npz"
        z = np.load(path)
        token_ids, logits = z["token_ids"], z["logits"]
        refs_all = by_shard[shard]
        if len(refs_all) > args.per_shard:
            step = len(refs_all) / args.per_shard
            refs = [refs_all[int(i * step)] for i in range(args.per_shard)]
        else:
            refs = refs_all
        for ref in refs:
            row, lr = int(ref["row"]), int(ref["logit_rows"])
            L = min(lr, args.max_len)
            if L < 16:
                continue
            tids = token_ids[row:row + L]          # [L, K+1]
            tlps = logits[row:row + L]             # [L, K+1]
            actual_ids = tids[:, 0].astype(np.int64)
            inp = prefix + [int(x) for x in actual_ids[: L]]
            try:
                s_act, s_top = req_logprobs(args.api, args.model, inp, args.topk, args.timeout)
            except Exception as e:
                print(f"    [warn] {ref['id']} request failed: {e}", flush=True)
                continue
            off = len(prefix)          # student prompt position i+off <-> teacher row i
            ce_gap = 0.0
            ag1 = 0
            ov5 = 0
            kl_sum = 0.0
            n = 0
            for i in range(L):
                p = i + off
                if p >= len(s_act) or s_act[p] is None:
                    continue
                s_alp = s_act[p]
                if s_alp is None:
                    continue
                writer = s_top[p] or {}
                s_lp_map = {int(str(k).split(":")[-1]): float(v) for k, v in writer.items()}
                ps_map = {k: math.exp(v) for k, v in s_lp_map.items()}
                if not ps_map:
                    continue
                # teacher candidates: dedup {id: max lp} over col-0 (actual) + top-K
                td = {int(tids[i, 0]): float(tlps[i, 0])}
                for j in range(1, args.topk):
                    tid = int(tids[i, j]); lp = float(tlps[i, j])
                    if lp > td.get(tid, -1e9):
                        td[tid] = lp
                t_sorted = sorted(td.items(), key=lambda kv: -kv[1])
                t_cand_id = np.asarray([k for k, _ in t_sorted], dtype=np.int64)
                t_cand_lp = np.asarray([v for _, v in t_sorted], dtype=np.float64)
                t_top1_id, t_top1_lp = t_sorted[0]
                t_top2_lp = t_sorted[1][1] if len(t_sorted) > 1 else t_top1_lp
                # student top-1/2 (margins are log-prob differences)
                s_sorted = sorted(s_lp_map.items(), key=lambda kv: -kv[1])
                s_top1_id, s_top1_lp = s_sorted[0]
                s_top2_lp = s_sorted[1][1] if len(s_sorted) > 1 else s_top1_lp
                ce_gap += float(tlps[i, 0]) - float(s_alp)
                ag1 += int(int(t_top1_id) == s_top1_id)
                ov5 += len(set(int(x) for x in t_cand_id[:5]) & set(k for k, _ in s_sorted[:5]))
                kl_sum += kl_topk(t_cand_id, t_cand_lp, ps_map, args.vocab, args.topk)
                mt_list.append(t_top1_lp - t_top2_lp)
                ms_list.append(s_top1_lp - s_top2_lp)
                n += 1
            if n:
                n_pos += n
                agg["ce_gap"] += ce_gap
                agg["ag1"] += ag1
                agg["ov5"] += ov5
                agg["kl"] += kl_sum
                per_sample.append({"id": ref["id"], "n": n, "ce_gap": ce_gap / n,
                                   "ag1": ag1 / n, "ov5": ov5 / n, "kl": kl_sum / n})
        print(f"[fid] {shard} done, positions so far={n_pos} ({time.time()-t_start:.0f}s)", flush=True)

    if not n_pos:
        print("[fid] no positions collected", file=sys.stderr)
        sys.exit(1)
    res = {
        "model": args.model, "n_positions": n_pos,
        "nll_gap_mean": agg["ce_gap"] / n_pos,            # NLL(student)-NLL(teacher)
        "ppl_ratio": math.exp(agg["ce_gap"] / n_pos),
        "top1_agreement": agg["ag1"] / n_pos,
        "top5_overlap_mean": agg["ov5"] / n_pos,
        "kl_topk_proxy_mean": agg["kl"] / n_pos,
        "margin_coef_c": None, "margin_intercept_b": None,
        "per_sample": per_sample,
    }
    mt = np.asarray(mt_list, dtype=np.float64)
    ms = np.asarray(ms_list, dtype=np.float64)
    if mt.size > 10:
        c, b = np.polyfit(mt, ms, 1)
        res["margin_coef_c"] = float(c)
        res["margin_intercept_b"] = float(b)
        res["margin_mt_q"] = [float(x) for x in np.quantile(mt, [0.1, 0.5, 0.9, 0.99])]
        res["margin_ms_q"] = [float(x) for x in np.quantile(ms, [0.1, 0.5, 0.9, 0.99])]
        res["margin_corr"] = float(np.corrcoef(mt, ms)[0, 1])
        # decision-like subset: teacher margin small (near-tie decisions)
        for thr in (2.0, 4.0, 6.0):
            m = mt <= thr
            if m.sum() > 20:
                cc, bb = np.polyfit(mt[m], ms[m], 1)
                res[f"margin_c_thr{int(thr)}"] = float(cc)
                res[f"margin_b_thr{int(thr)}"] = float(bb)
                res[f"margin_frac_thr{int(thr)}"] = float(m.mean())
    print("\n============= DISTRIBUTION FIDELITY =============")
    for k, v in res.items():
        if k != "per_sample":
            print(f"  {k:24}: {v}")
    print("=================================================")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[fid] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
