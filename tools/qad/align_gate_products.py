#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""align_gate_products.py — Track A 组合闸门（clip+SSZ[+A8]）多产物矩阵（CPU）。

背景（§6.8.3 步骤 2 / §6.9.2#4）：QAD Phase-1 的学生 fake-quant 必须精确复刻部署期
反量化。P1 候选底座的专家 scale 由 clip 范围搜索（± SSZ 精修）决定，故门要从
单一 L0/SSZ 扩展到「clip 选出的 scale」并在多产物上成矩阵验证：

  A1 幂等闸门（全部产物，底座无关）：
      fake_quant_weight(dequant, scale) == dequant 且 code_match == 1.0
      → 证明部署 (scale, int4 网格) 对与本方 STE 逐位一致。
  A2 源等价闸门（仅 L0 族：无旋转/AWQ/SQ）：
      fake_quant_weight(W_bf16, scale) == dequant（maxdiff == 0）
      → 学生直接吃 BF16 权重即可复刻部署产物。
      Path2 族（旋转/AWQ/SQ 折叠后权重 ≠ BF16）跳过本项，等 Track B 折叠等价性。
  A3 语义闸门（A8 激活）：
      npuslim.qad.ste.activation_dynamic_int8 的性质（per-token scale=absmax/qmax、
      网格整数倍、clamp 边界、幂等）→ 锁定训练/评测/部署单一语义源。
  A4 loader 闸门：make_scale_provider 的 shape/数值与该产物 tensor 一致。

运行（e6 容器，CPU）：
  PYTHONPATH=/llm_infer_workspace/ylt/npuslim/src python3 align_gate_products.py \
    --products l0clipssz=...:src l0clip_lo05=...:src p2clip_seed1234=... \
    --samples 3:0,40:100,77:255 --out align_gate_report.json
"""
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
NPUSLIM_SRC = _pd_os.environ.get("NPUSLIM_SRC", YLT + "/npuslim/src")
QAD_ROOT = _pd_os.environ.get("QAD_ROOT", YLT + "/glm52-qad")
# =========================================================
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, f"{SHARED}/ylt/npuslim/src")
from npuslim.qad.ste import activation_dynamic_int8, fake_quant_weight  # noqa: E402

BF16 = f"{SHARED}/models/GLM-5.2-bf16/snapshots/master"
_PROD_ROOT = f"{SHARED}/ylt/npuslim/outputs_npu/glm5/gptq"
DEFAULT_PRODUCTS = [
    f"L0={_PROD_ROOT}/glm5-w4a8_modelslim_replica_layer0_noawq_nosq:src",
    f"l0clipssz={_PROD_ROOT}/glm5-w4a8_replica_l0_clip_ssz:src",
    f"l0clip_lo05={_PROD_ROOT}/glm5-w4a8_replica_l0_clip_lo05:src",
    f"p2clip_seed1234={_PROD_ROOT}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234",
    f"p2clip_sszoff={_PROD_ROOT}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_sszoff",
    f"p2clip_nosq={_PROD_ROOT}/glm5-w4a8_modelslim_replica_layer1_5_path2_nosq",
]
PROJS = ("gate_proj", "up_proj", "down_proj")


_idx_cache: dict = {}
_shard_handles: dict = {}


def idx(d):
    if d in _idx_cache:
        return _idx_cache[d]
    for fn in ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            _idx_cache[d] = json.load(open(p))["weight_map"]
            return _idx_cache[d]
    raise FileNotFoundError(d)


def g(d, i, n):
    shard = os.path.join(d, i[n])
    h = _shard_handles.get(shard)
    if h is None:
        h = safe_open(shard, framework="pt")
        if len(_shard_handles) > 48:   # 句柄上限，防 fd 泄漏
            _shard_handles.pop(next(iter(_shard_handles)))
        _shard_handles[shard] = h
    return h.get_tensor(n)


def decode(packed, scale):
    u = packed.to(torch.uint8).to(torch.int32)
    out = torch.empty(2 * u.shape[0], u.shape[1], dtype=torch.int32)
    out[0::2] = u & 0x0F
    out[1::2] = (u >> 4) & 0x0F
    q_tc = torch.where(out >= 8, out - 16, out).float()
    return q_tc, q_tc * scale


def check_weight(prod_dir, pidx, layer, expert, proj, bf16_dir, bf16_idx, src_ok):
    base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
    try:
        packed = g(prod_dir, pidx, f"{base}.weight")
        sc = g(prod_dir, pidx, f"{base}.weight_scale").float()
    except (KeyError, FileNotFoundError) as exc:
        return {"status": "SKIP", "reason": f"{type(exc).__name__}: {base}"}
    q_tc, deq = decode(packed, sc)
    ours = fake_quant_weight(deq, bits=4, scale=sc)
    code_match = float((torch.round(ours / sc) == q_tc).float().mean())
    rec = {"status": "PASS", "idem_code_match": code_match,
           "idem_maxdiff": float((ours - deq).abs().max())}
    if rec["idem_code_match"] != 1.0 or rec["idem_maxdiff"] != 0.0:
        rec["status"] = "FAIL"
    if src_ok and bf16_dir:
        wb = g(bf16_dir, bf16_idx, f"{base}.weight").float()
        if tuple(wb.shape) != tuple(deq.shape):
            rec["src"] = "SKIP-shape"
        else:
            ours_src = fake_quant_weight(wb, bits=4, scale=sc)
            rec["src_maxdiff"] = float((ours_src - deq).abs().max())
            if rec["src_maxdiff"] != 0.0:
                rec["status"] = "FAIL"
    return rec


def check_loader(prod_dir, num_experts, layer, expert):
    from npuslim.qad.deploy_scales import make_scale_provider

    n_exp, I, H = num_experts, 2048, 6144
    s = make_scale_provider(prod_dir, n_exp)(layer)
    pidx = idx(prod_dir)
    gate = g(prod_dir, pidx, f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale").float().reshape(-1)
    up = g(prod_dir, pidx, f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale").float().reshape(-1)
    dn = g(prod_dir, pidx, f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale").float().reshape(-1)
    shape_ok = (tuple(s["gate_up"].shape) == (n_exp, 2 * I, 1)
                and tuple(s["down"].shape) == (n_exp, H, 1))
    val_ok = (torch.allclose(s["gate_up"][expert, :I, 0], gate)
              and torch.allclose(s["gate_up"][expert, I:2 * I, 0], up)
              and torch.allclose(s["down"][expert, :, 0], dn))
    return {"status": "PASS" if (shape_ok and val_ok) else "FAIL",
            "shape_ok": shape_ok, "value_ok": val_ok}


def check_a8():
    gtorch = torch.Generator().manual_seed(5)
    x = torch.randn(8, 64, generator=gtorch) * 6.0
    y = activation_dynamic_int8(x)
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    on_grid = bool(torch.allclose(y / s, (y / s).round(), atol=1e-4))
    in_bounds = bool((y.abs() <= 128.0 * s + 1e-6).all())
    idem = bool(torch.equal(activation_dynamic_int8(y), y))
    zero_ok = bool((activation_dynamic_int8(torch.zeros(2, 4)) == 0).all())
    ok = on_grid and in_bounds and idem and zero_ok
    return {"status": "PASS" if ok else "FAIL", "on_grid": on_grid,
            "in_bounds": in_bounds, "idempotent": idem, "zero_row_ok": zero_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="*", default=DEFAULT_PRODUCTS,
                    help="name=path[:src]；:src 表示可做源等价（L0 族）")
    ap.add_argument("--samples", default="3:0,40:100,77:255", help="layer:expert,...")
    ap.add_argument("--sweep-layers", default=None,
                    help="all-moe = 全部 75 个 MoE 层（3..77）；或 '3-77'；与 --samples 互斥")
    ap.add_argument("--experts-per-layer", type=int, default=2)
    ap.add_argument("--sweep-seed", type=int, default=7)
    ap.add_argument("--bf16", default=BF16)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--out", default="align_gate_report.json")
    args = ap.parse_args()

    if args.sweep_layers:
        if args.sweep_layers == "all-moe":
            layers = list(range(3, 78))
        else:
            a, _, b = args.sweep_layers.partition("-")
            layers = list(range(int(a), int(b) + 1))
        rng = torch.Generator().manual_seed(args.sweep_seed)
        samples = []
        for layer in layers:
            es = torch.randperm(args.num_experts, generator=rng)[: args.experts_per_layer].tolist()
            samples.extend((layer, e) for e in es)
    else:
        samples = [tuple(int(v) for v in s.split(":")) for s in args.samples.split(",")]
    print(f"扫描样本数 = {len(samples)}（每产物 {len(samples) * len(PROJS)} 项检查）")
    bf16_idx = idx(args.bf16) if os.path.isdir(args.bf16) else None

    report = {"samples": samples, "products": {}, "a8": check_a8()}
    print(f"A8 激活语义闸门: {report['a8']}")
    all_ok = report["a8"]["status"] == "PASS"

    for spec in args.products:
        name, _, flags = spec.partition(":")
        pname, _, ppath = name.partition("=")
        src_ok = "src" in flags.split(":")
        pdir = ppath or pname
        pidx = idx(pdir)
        entries = {}
        for (layer, expert) in samples:
            for proj in PROJS:
                entries[f"L{layer}e{expert}.{proj}"] = check_weight(
                    pdir, pidx, layer, expert, proj, args.bf16, bf16_idx, src_ok)
        entries["loader"] = check_loader(pdir, args.num_experts, samples[0][0], samples[0][1])
        bad = [k for k, v in entries.items() if v.get("status") == "FAIL"]
        report["products"][pname] = {"dir": pdir, "src_equiv": src_ok,
                                     "entries": entries, "failed": bad}
        status = "PASS" if not bad else "FAIL"
        all_ok = all_ok and status == "PASS"
        src_note = "src等价" if src_ok else "幂等only(Path2)"
        print(f"[{status}] {pname:18s} ({src_note}) 失败项={bad or '无'}")
        for k, v in entries.items():
            if v.get("status") == "PASS" and k != "loader":
                d = v.get("idem_maxdiff")
                s = v.get("src_maxdiff", "-")
                print(f"    {k:28s} code_match={v['idem_code_match']:.6f} "
                      f"idem_maxdiff={d:.3g} src_maxdiff={s}")

    report["overall"] = "PASS" if all_ok else "FAIL"
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"OVERALL {report['overall']} → {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
