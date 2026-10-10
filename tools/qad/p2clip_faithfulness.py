#!/usr/bin/env python
"""p2clip seed1234 scale faithfulness check (Phase 4 side-check).

Compares for a few (layer, expert, proj):
  LHS = fake_quant_weight(bf16_W_from_MODEL_DIR, scale_from_artifact)
        — STE symmetric: clamp(round(W/s), -8, 7) * s
  RHS = unpack_int4(weight_artifact) * weight_scale + weight_offset
        — artifact's stored dequantized weight (W4A8 packed int4 → dequant)

W4A8 packing (npuslim gptq_algo.py::_pack_ascend / msmodelslim pack.py::w4a8_pack_int4):
  pairs of output rows share one int8 byte; even row → low nibble, odd → high nibble;
  int4 values are two's complement [-8,7] stored as unsigned nibble [0,15].
"""
from __future__ import annotations

import os
import sys
import json

import torch
from safetensors import safe_open

sys.path.insert(0, "/llm_infer_workspace/ylt/npuslim/src")
from npuslim.qad.ste import fake_quant_weight

MODEL_DIR = "/llm_infer_workspace/models/GLM-5.2-bf16/snapshots/master"
ARTIFACT_DIR = "/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234"
CURRENT_ART = "/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_ssz"


def load_index(artifact_dir):
    p = os.path.join(artifact_dir, "model.safetensors.index.json")
    with open(p) as f:
        return json.load(f)["weight_map"]


def load_tensor(d, wm, name):
    with safe_open(os.path.join(d, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def unpack_int4(packed_int8: torch.Tensor) -> torch.Tensor:
    """Unpack W4A8 row-wise packed int4 from int8 [out//2, in] → signed int4 [out, in]."""
    u8 = (packed_int8.to(torch.int32) & 0xFF).to(torch.uint8)
    low = (u8 & 0x0F).to(torch.int32)
    high = ((u8 >> 4) & 0x0F).to(torch.int32)
    low_signed = low - 16 * (low >= 8).to(torch.int32)
    high_signed = high - 16 * (high >= 8).to(torch.int32)
    out = torch.empty(packed_int8.shape[0] * 2, packed_int8.shape[1], dtype=torch.int32)
    out[0::2] = low_signed
    out[1::2] = high_signed
    return out


def check_one(model_dir, model_wm, art_dir, art_wm, layer, expert, proj):
    w_key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    s_key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight_scale"
    o_key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight_offset"

    bf16_W = load_tensor(model_dir, model_wm, w_key).float()
    scale = load_tensor(art_dir, art_wm, s_key).float()
    offset = load_tensor(art_dir, art_wm, o_key).float()
    q_weight = load_tensor(art_dir, art_wm, w_key)

    if q_weight.dtype == torch.int8 and q_weight.shape[0] * 2 == bf16_W.shape[0]:
        q_int = unpack_int4(q_weight).float()
    else:
        q_int = q_weight.float()

    scale = scale.reshape(-1, 1)
    offset = offset.reshape(-1, 1)

    lhs = fake_quant_weight(bf16_W, bits=4, scale=scale)
    rhs = q_int * scale + offset

    maxdiff = (lhs - rhs).abs().max().item()
    offset_max = offset.abs().max().item()
    q_range = (q_int.min().item(), q_int.max().item())

    q_ste = torch.clamp(torch.round(bf16_W / scale), -8, 7)
    int_match = (q_ste == q_int).float().mean().item()
    int_maxdiff = (q_ste - q_int).abs().max().item()

    print(f"  L{layer} E{expert} {proj}: maxdiff={maxdiff:.6g} "
          f"offset_max={offset_max:.6g} q_range=[{q_range[0]:.0f},{q_range[1]:.0f}]")
    print(f"    int grid match: {int_match*100:.1f}% (int_maxdiff={int_maxdiff:.0f})")
    return maxdiff, int_match, offset_max


def main():
    model_wm = load_index(MODEL_DIR)

    print("=" * 70)
    print("p2clip seed1234 faithfulness check")
    print(f"  MODEL_DIR:    {MODEL_DIR}")
    print(f"  ARTIFACT_DIR: {ARTIFACT_DIR}")
    print("=" * 70)

    art_wm = load_index(ARTIFACT_DIR)

    checks = [
        (5, 0, "gate_proj"), (5, 0, "down_proj"),
        (5, 127, "gate_proj"), (5, 255, "down_proj"),
        (10, 0, "gate_proj"), (10, 50, "down_proj"),
        (40, 100, "gate_proj"), (40, 200, "down_proj"),
        (70, 0, "up_proj"), (70, 255, "down_proj"),
    ]

    all_md, all_im, all_om = [], [], []
    for layer, expert, proj in checks:
        try:
            md, im, om = check_one(MODEL_DIR, model_wm, ARTIFACT_DIR, art_wm,
                                   layer, expert, proj)
            all_md.append(md); all_im.append(im); all_om.append(om)
        except Exception as e:
            print(f"  L{layer} E{expert} {proj}: ERROR {e}")

    print()
    print(f"=== p2clip seed1234 summary ===")
    print(f"  maxdiff range: [{min(all_md):.6g}, {max(all_md):.6g}]")
    print(f"  int grid match: [{min(all_im)*100:.1f}%, {max(all_im)*100:.1f}%]")
    print(f"  offset_max range: [{min(all_om):.6g}, {max(all_om):.6g}]")
    if max(all_md) < 1e-4:
        print("  VERDICT: drop-in swap (maxdiff ~ 0)")
    elif max(all_om) < 1e-7 and min(all_im) > 0.999:
        print("  VERDICT: drop-in swap (symmetric, int grid matches, float rounding only)")
    else:
        print("  VERDICT: training must first apply same weight transform (clip/rotate)")

    print()
    print("=" * 70)
    print(f"CURRENT deploy-scale artifact (l0_clip_ssz) for comparison")
    print(f"  ARTIFACT_DIR: {CURRENT_ART}")
    print("=" * 70)
    try:
        cur_wm = load_index(CURRENT_ART)
        cur_md, cur_im = [], []
        for layer, expert, proj in checks[:4]:
            try:
                md, im, om = check_one(MODEL_DIR, model_wm, CURRENT_ART, cur_wm,
                                       layer, expert, proj)
                cur_md.append(md); cur_im.append(im)
            except Exception as e:
                print(f"  L{layer} E{expert} {proj}: ERROR {e}")
        if cur_md:
            print(f"  current summary: maxdiff=[{min(cur_md):.6g}, {max(cur_md):.6g}] "
                  f"int_match=[{min(cur_im)*100:.1f}%, {max(cur_im)*100:.1f}%]")
    except Exception as e:
        print(f"  CURRENT artifact check failed: {e}")


if __name__ == "__main__":
    main()
