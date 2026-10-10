#!/usr/bin/env python3
"""diag_flip_stats.py v4 — A0-1: 离线翻转码统计（真正流式，B@A on the fly）

策略：加载 ckpt 一次（11GB），然后逐 shard 流式处理底座；每个专家的
B@A 按需即时计算不缓存。防 OOM。
"""

from __future__ import annotations
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")
NPUSLIM_SRC = _pd_os.environ.get("NPUSLIM_SRC", YLT + "/npuslim/src")
QAD_ROOT = _pd_os.environ.get("QAD_ROOT", YLT + "/glm52-qad")
# =========================================================

import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open

LO_CLIP = f"{SHARED}/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni"
CKPT_DIR = f"{SHARED}/ylt/glm52-qad/ckpt_e2"
SCALING = 4.0
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

# Constant shapes: LoRA B (2048|6144, 16), A (16, 6144|2048)
GATE_UP_SHAPE = (2048, 6144)  # after B@A
DOWN_SHAPE = (6144, 2048)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    u8 = (p.to(torch.int32) & 0xFF).to(torch.uint8)
    lo = (u8 & 0x0F).to(torch.int32)
    hi = ((u8 >> 4) & 0x0F).to(torch.int32)
    lo = lo - 16 * (lo >= 8).to(torch.int32)
    hi = hi - 16 * (hi >= 8).to(torch.int32)
    out = torch.empty(p.shape[0] * 2, p.shape[1], dtype=torch.int32)
    out[0::2], out[1::2] = lo, hi
    return out


def process_ckpt(ckpt_path: str, label: str, wm: dict, shard_list: list[str]) -> dict:
    """Load ckpt, stream base shards, compute B@A on demand."""
    t0 = time.time()
    print(f"\n── {label} ── ({time.strftime('%H:%M:%S')})", flush=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 保留 LoRA 参数张量，B@A 按需计算（不预存 58k 结果）
    print(f"  ckpt loaded ({time.time()-t0:.1f}s)", flush=True)

    total_flip = 0
    total_elems = 0
    d_means: list[float] = []
    d_rms: list[float] = []
    proj_flip = {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
    proj_elems = {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
    band_flip = {"L0-25": 0, "L26-50": 0, "L51-74": 0}
    band_elems = {"L0-25": 0, "L26-50": 0, "L51-74": 0}

    n_processed = 0
    for sf in shard_list:
        with safe_open(f"{LO_CLIP}/{sf}", framework="pt", device="cpu") as fd:
            # 先读取该 shard 包含的所有专家权重名称
            keys_in_shard = list(fd.keys())
            for name in keys_in_shard:
                m = EXPERT_RE.match(name)
                if not m:
                    continue
                layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)

                # 按需计算 B@A（如果该层有 LoRA 参数，否则 δ=0 无翻转）
                p = f"model.layers.{layer}.mlp.experts.lora.params"
                try:
                    if proj in ("gate_proj", "up_proj"):
                        bgu = sd[f"{p}.Bgu"][expert].float()
                        agu = sd[f"{p}.Agu"][expert].float()
                        d = (bgu @ agu) * SCALING
                        half = d.shape[0] // 2
                        d = d[:half] if proj == "gate_proj" else d[half:]
                    else:  # down_proj
                        bdn = sd[f"{p}.Bdn"][expert].float()
                        adn = sd[f"{p}.Adn"][expert].float()
                        d = (bdn @ adn) * SCALING
                except (KeyError, IndexError):
                    # 该层无该专家的 LoRA（如 dense 层或 expert 数不齐）→ δ=0 → 无翻转
                    continue

                # 加载底座码本和 scale（同 shard，已验证）
                q_base = fd.get_tensor(name)
                s_name = name.replace(".weight", ".weight_scale")
                s_scale = fd.get_tensor(s_name).float().reshape(-1, 1)

                q_signed = unpack_int4(q_base).float()
                deq = q_signed * s_scale
                wp = deq + d
                q_new = torch.clamp(torch.round(wp / s_scale), -8, 7)
                q_old = q_signed.to(q_new.dtype)

                flip = (q_new != q_old)
                n_flip = int(flip.sum().item())
                n_elems = flip.numel()

                total_flip += n_flip
                total_elems += n_elems
                proj_flip[proj] += n_flip
                proj_elems[proj] += n_elems
                band_key = "L0-25" if layer <= 25 else "L26-50" if layer <= 50 else "L51-74"
                band_flip[band_key] += n_flip
                band_elems[band_key] += n_elems

                d_means.append(d.abs().mean().item())
                d_rms.append(d.pow(2).mean().sqrt().item())

                n_processed += 1
                if n_processed % 5000 == 0:
                    print(f"    processed {n_processed}/{len(d_means)} "
                          f"(翻转累计 {total_flip:,}) ...", flush=True)

    elapsed = time.time() - t0
    ratio = total_flip / total_elems * 100 if total_elems else 0
    d_means.sort()
    d_rms.sort()

    print(f"  完成: {elapsed:.1f}s, "
          f"flip={total_flip:,}/{total_elems:,} ({ratio:.3f}%), "
          f"δ|mean|med={d_means[len(d_means)//2]:.3e}, "
          f"δ RMS med={d_rms[len(d_rms)//2]:.3e}", flush=True)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pr = proj_flip[proj] / proj_elems[proj] * 100 if proj_elems[proj] else 0
        print(f"    {proj:<12}: {proj_flip[proj]:>8,}/{proj_elems[proj]:>8,} ({pr:.2f}%)", flush=True)
    for band in ("L0-25", "L26-50", "L51-74"):
        br = band_flip[band] / band_elems[band] * 100 if band_elems[band] else 0
        print(f"    {band:<8}: {band_flip[band]:>8,}/{band_elems[band]:>8,} ({br:.2f}%)", flush=True)

    del sd
    return {"step": label, "flip": total_flip, "elems": total_elems,
            "ratio": ratio, "d_mean_med": d_means[len(d_means)//2],
            "d_rms_med": d_rms[len(d_rms)//2]}


def main() -> None:
    ts0 = time.time()
    print("=" * 80, flush=True)
    print(f"A0-1: run14 离线翻转码统计", flush=True)
    print(f"底座: {LO_CLIP.split('/')[-1]}", flush=True)
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    with open(f"{LO_CLIP}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    print(f"底座张量总数: {len(wm)}", flush=True)

    # 含专家权重的 shard
    shard_sets: dict[str, set] = {}
    for name in wm:
        m = EXPERT_RE.match(name)
        if m:
            shard_sets.setdefault(wm[name], set()).add(name)
    shard_list = sorted(shard_sets.keys())
    print(f"含专家权重的 shard: {len(shard_list)}", flush=True)

    steps = [200, 400, 600, 800, 1000, 1200, 1400]
    results = []
    for s in steps:
        ckpt_path = f"{CKPT_DIR}/lora_step{s}_p.pt"
        if not os.path.exists(ckpt_path):
            print(f"\n⚠ 跳过不存在: {ckpt_path}", flush=True)
            continue
        r = process_ckpt(ckpt_path, str(s), wm, shard_list)
        results.append(r)

    # 汇总
    print("\n" + "=" * 80, flush=True)
    print(f"汇总表 (底座: {LO_CLIP.split('/')[-1]})", flush=True)
    print(f"{'Step':>6} {'翻转数':>12} {'翻转占比':>10} {'δ|mean|median':>14} {'δ RMS median':>14}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        print(f"{r['step']:>6} {r['flip']:>12,} {r['ratio']:>9.3f}% "
              f"{r['d_mean_med']:>14.3e} {r['d_rms_med']:>14.3e}", flush=True)
    print(f"\n总耗时: {time.time()-ts0:.1f}s", flush=True)
    print(f"\n（run12 qad600 历史口径：翻转=2,920,258, δ|mean| med=1.05e-6）", flush=True)
    print(f"（注意：run14 底座 l0 ≠ run12 底座 p2clip_s1234，不可直接比）", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
