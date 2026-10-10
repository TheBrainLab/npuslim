#!/usr/bin/env python
"""B2 artifact loader correctness validation (CPU, runs on node).

For ~10 (layer, expert, proj) samples + non-expert quantized keys + float-passthrough
keys: asserts the loader's _dequant_artifact_weight output equals an INDEPENDENT
reference dequant (unpack_int4 * scale + offset computed inline), maxdiff == 0.

Usage (inside container):
  /llm_infer_workspace/ylt/venv_t512/bin/python \
    /llm_infer_workspace/ylt/glm52-qad/validate_artifact_loader.py
"""
from __future__ import annotations

import os
import sys
import json
import collections

import torch
from safetensors import safe_open

sys.path.insert(0, "/llm_infer_workspace/ylt/glm52-qad")
from stream_load_checkpoint import unpack_int4, _dequant_artifact_weight  # noqa: E402

ARTIFACT_DIR = "/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234"


def load_index(d):
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load_tensor(d, wm, name):
    with safe_open(os.path.join(d, wm[name]), framework="pt", device="cpu") as sf:
        return sf.get_tensor(name)


def ref_unpack_dequant(weight, scale, offset):
    """Independent reference: inline unpack + dequant (not calling loader's function)."""
    if weight.dtype != torch.int8:
        return weight.float()
    s = scale.float().reshape(-1, 1)
    o = offset.float().reshape(-1, 1)
    if s.shape[0] == 2 * weight.shape[0]:
        u8 = (weight.to(torch.int32) & 0xFF).to(torch.uint8)
        low = (u8 & 0x0F).to(torch.int32)
        high = ((u8 >> 4) & 0x0F).to(torch.int32)
        low_s = low - 16 * (low >= 8).to(torch.int32)
        high_s = high - 16 * (high >= 8).to(torch.int32)
        q = torch.empty(weight.shape[0] * 2, weight.shape[1], dtype=torch.int32)
        q[0::2] = low_s
        q[1::2] = high_s
        q = q.float()
    else:
        q = weight.to(torch.int32).float()
    return q * s + o


def check_key(wm, key, label):
    weight = load_tensor(ARTIFACT_DIR, wm, key)
    if weight.dtype == torch.int8:
        scale_key = key + "_scale"
        offset_key = key + "_offset"
        if scale_key not in wm or offset_key not in wm:
            print(f"  [{label}] {key}: SKIP (no scale/offset)")
            return None
        scale = load_tensor(ARTIFACT_DIR, wm, scale_key)
        offset = load_tensor(ARTIFACT_DIR, wm, offset_key)
        ref = ref_unpack_dequant(weight, scale, offset)
        loader = _dequant_artifact_weight(weight, scale, offset)
        md = (ref - loader).abs().max().item()
        print(f"  [{label}] {key}: dtype={weight.dtype} shape={tuple(weight.shape)} "
              f"-> dequant {tuple(loader.shape)} maxdiff={md:.2e} "
              f"{'OK' if md == 0 else 'FAIL'}")
        del weight, scale, offset
        return md
    else:
        loader = _dequant_artifact_weight(weight, torch.empty(0), torch.empty(0))
        md = (weight.float() - loader.float()).abs().max().item()
        print(f"  [{label}] {key}: dtype={weight.dtype} shape={tuple(weight.shape)} "
              f"passthrough maxdiff={md:.2e} {'OK' if md == 0 else 'FAIL'}")
        del weight
        return md


def main():
    wm = load_index(ARTIFACT_DIR)
    print("=" * 70)
    print(f"B2 artifact loader correctness validation")
    print(f"  ARTIFACT_DIR: {ARTIFACT_DIR}")
    print(f"  total keys: {len(wm)}")
    print("=" * 70)

    # dtype histogram (sample first 2000 keys for speed)
    dt_hist = collections.Counter()
    for k in list(wm.keys())[:2000]:
        t = load_tensor(ARTIFACT_DIR, wm, k)
        dt_hist[str(t.dtype)] += 1
        del t
    print(f"\n[dtype histogram] (sample 2000 keys): {dict(dt_hist)}")

    all_md = []

    print("\n--- Expert quantized weights (10 samples, layers 3-74) ---")
    expert_samples = [
        (3, 0, "gate_proj"), (3, 100, "down_proj"), (3, 255, "up_proj"),
        (10, 0, "down_proj"), (10, 127, "gate_proj"),
        (40, 50, "up_proj"), (40, 200, "down_proj"),
        (60, 0, "gate_proj"), (60, 255, "down_proj"),
        (74, 128, "up_proj"),
    ]
    for layer, expert, proj in expert_samples:
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        if key not in wm:
            print(f"  [expert] {key}: NOT FOUND")
            continue
        md = check_key(wm, key, "expert")
        if md is not None:
            all_md.append(md)

    print("\n--- Non-expert quantized weights (attn/dense) ---")
    nonexpert_samples = [
        "model.layers.5.self_attn.q_a_proj.weight",
        "model.layers.5.self_attn.q_b_proj.weight",
        "model.layers.5.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.5.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.indexer.wq_b.weight",
    ]
    for key in nonexpert_samples:
        if key not in wm:
            print(f"  [non-expert] {key}: NOT FOUND")
            continue
        md = check_key(wm, key, "non-expert")
        if md is not None:
            all_md.append(md)

    print("\n--- Float passthrough keys (3 samples) ---")
    passthrough_samples = [
        "model.layers.5.input_layernorm.weight",
        "model.layers.5.mlp.gate.weight",
        "model.layers.5.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        "model.layers.78.eh_proj.weight",
    ]
    for key in passthrough_samples:
        if key not in wm:
            print(f"  [passthrough] {key}: NOT FOUND")
            continue
        md = check_key(wm, key, "passthrough")
        if md is not None:
            all_md.append(md)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_md)} checks, max maxdiff={max(all_md):.2e}")
    if max(all_md) == 0:
        print("VERDICT: ALL PASS (maxdiff == 0)")
    else:
        print("VERDICT: FAIL (nonzero maxdiff)")
    print("=" * 70)


if __name__ == "__main__":
    main()
