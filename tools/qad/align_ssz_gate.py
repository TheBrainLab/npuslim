#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""align_ssz_gate.py — QAD Phase-1 SSZ-scale 接线对齐门（CPU）。

验证：STE fake_quant(W_bf16, scale=产物 weight_scale) == 产物反量化（解码 int4 packed × scale）。
运行（e6 容器）：
  PYTHONPATH=/llm_infer_workspace/ylt/npuslim/src python3 align_ssz_gate.py
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/llm_infer_workspace/ylt/npuslim/src")
from npuslim.qad.ste import fake_quant_weight  # noqa: E402

BF16 = "/llm_infer_workspace/models/GLM-5.2-bf16/snapshots/master"
L0 = "/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer0_noawq_nosq"


def idx(d):
    for fn in ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return json.load(open(p))["weight_map"]
    raise FileNotFoundError(d)


def g(d, i, n):
    with safe_open(os.path.join(d, i[n]), framework="pt") as f:
        return f.get_tensor(n)


def decode(packed, scale):
    u = packed.to(torch.uint8).to(torch.int32)
    out = torch.empty(2 * u.shape[0], u.shape[1], dtype=torch.int32)
    out[0::2] = u & 0x0F
    out[1::2] = (u >> 4) & 0x0F
    q_tc = torch.where(out >= 8, out - 16, out).float()
    return q_tc, q_tc * scale


def main():
    bi, li = idx(BF16), idx(L0)
    L, e = 3, 0
    for proj in ("gate_proj", "up_proj", "down_proj"):
        wb = g(BF16, bi, f"model.layers.{L}.mlp.experts.{e}.{proj}.weight").float()
        packed = g(L0, li, f"model.layers.{L}.mlp.experts.{e}.{proj}.weight")
        sc = g(L0, li, f"model.layers.{L}.mlp.experts.{e}.{proj}.weight_scale").float()
        q_tc, deq = decode(packed, sc)
        ours = fake_quant_weight(wb, bits=4, scale=sc)
        d = (deq - ours).abs()
        ours_q = torch.round(ours / sc)
        code_eq = (ours_q == q_tc).float().mean()
        print(f"  {proj:9s} shape={tuple(wb.shape)} maxdiff={float(d.max()):.4g} "
              f"mean|d|={float(d.mean()):.4g} code_match={float(code_eq):.6f}")

    from npuslim.qad.deploy_scales import make_scale_provider
    num_experts, I, H = 256, 2048, 6144
    s = make_scale_provider(L0, num_experts)(L)
    gate_sc = g(L0, li, f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale").float().reshape(-1)
    up_sc = g(L0, li, f"model.layers.{L}.mlp.experts.{e}.up_proj.weight_scale").float().reshape(-1)
    dn_sc = g(L0, li, f"model.layers.{L}.mlp.experts.{e}.down_proj.weight_scale").float().reshape(-1)
    shape_ok = (tuple(s["gate_up"].shape) == (num_experts, 2 * I, 1)
                and tuple(s["down"].shape) == (num_experts, H, 1))
    gu_ok = (torch.allclose(s["gate_up"][e, :I, 0], gate_sc)
             and torch.allclose(s["gate_up"][e, I:2 * I, 0], up_sc))
    dn_ok = torch.allclose(s["down"][e, :, 0], dn_sc)
    print(f"  LOADER gate_up={tuple(s['gate_up'].shape)} down={tuple(s['down'].shape)} "
          f"shape_ok={shape_ok} gu_ok={gu_ok} dn_ok={dn_ok}")
    print("ALIGN GATE DONE")


if __name__ == "__main__":
    main()
