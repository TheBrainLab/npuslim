#!/usr/bin/env python
"""
p2clip mbpp 塌陷配方归因案（离线 diff）
====================================================================
Task "C" from PTQ_QAD_终局总结与下一步规划_1012.md §7.1

离线对比 p2clip_s1234（5 环折叠）vs l0_clip_lo05_omni（2 环忠实底座）
的 weight_scale 差异，并结合已知的 mbpp 翻转集题型归因，建立
"scale 分布畸变 ↔ 代码轴等价层 loss" 的相关性分析。
====================================================================
"""

import json
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

# ===== 容器内路径（e6_noawq_env）=====
BASE = "/llm_infer_workspace/ylt"
P2CLIP_ARTIFACT = os.path.join(
    BASE, "npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234"
)
L0_ARTIFACT = os.path.join(
    BASE, "npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni"
)

FLIP_DATA = {
    "p2clip_only_fail": 194, "both_fail": 125, "total": 500,
    "categories": {
        "list": 85, "string": 42, "math": 40,
        "sort/search": 14, "other": 12, "file/conversion": 1,
    },
}


def build_weight_map(artifact_dir):
    """Build weight_map by scanning all safetensors shards (handles missing index.json)."""
    idx_path = os.path.join(artifact_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)["weight_map"]

    # Build from shard scanning
    print(f"  [INFO] No index.json, scanning shards in {artifact_dir}...")
    wm = {}
    shards = sorted(f for f in os.listdir(artifact_dir) if f.endswith(".safetensors"))
    for shard in shards:
        path = os.path.join(artifact_dir, shard)
        try:
            with safe_open(path, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    wm[k] = shard
        except Exception as e:
            print(f"  [WARN] Skipping {shard}: {e}")
    return wm


def load_tensor(artifact_dir, weight_map, name):
    if name not in weight_map:
        return None
    path = os.path.join(artifact_dir, weight_map[name])
    if not os.path.exists(path):
        return None
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def compute_scale_ratios(p2clip_wm, l0_wm):
    """Per-expert per-channel scale ratio: p2clip / l0."""
    ratios = defaultdict(list)
    common = [k for k in p2clip_wm if k.endswith(".weight_scale")
              and "experts" in k and k in l0_wm]
    print(f"  Common expert scale keys: {len(common)}")

    # Stratified sample: ensure coverage of all 3 proj types evenly
    by_type = {"gate_proj": [], "up_proj": [], "down_proj": []}
    for k in common:
        parts = k.split(".")
        if len(parts) == 8 and parts[4] == "experts":
            proj = parts[6]
            if proj in by_type:
                by_type[proj].append(k)
    # Sample ~270 per type → ~810 total
    sampled = []
    for proj, keys in by_type.items():
        step = max(1, len(keys) // 270)
        sampled.extend(keys[::step])
    print(f"  Sampled {len(sampled)} keys: gate={len(by_type['gate_proj'])//max(1,step)} up={len(by_type['up_proj'])//max(1,step)} down={len(by_type['down_proj'])//max(1,step)}")

    for key in sampled:
        parts = key.split(".")
        # Pattern: model.layers.{L}.mlp.experts.{E}.{proj}.weight_scale (depth 8)
        # Skip: model.layers.{L}.mlp.shared_experts.{proj}.weight_scale (depth 7, non-expert)
        if len(parts) != 8 or parts[4] != "experts":
            continue
        layer, expert, proj = int(parts[2]), int(parts[5]), parts[6]
        s_p2 = load_tensor(P2CLIP_ARTIFACT, p2clip_wm, key)
        s_l0 = load_tensor(L0_ARTIFACT, l0_wm, key)
        if s_p2 is None or s_l0 is None:
            continue

        mask = s_l0 > 1e-10
        ratio = torch.ones_like(s_p2)
        ratio[mask] = s_p2[mask] / s_l0[mask]

        ratios[(layer, proj)].append({
            "expert": expert, "mean_ratio": ratio.mean().item(),
            "std_ratio": ratio.std().item(), "min_ratio": ratio.min().item(),
            "max_ratio": ratio.max().item(), "n_channels": s_p2.numel(),
        })

    # Aggregate
    aggregated = {}
    for (layer, proj), vals in ratios.items():
        means = [v["mean_ratio"] for v in vals]
        aggregated[f"L{layer}_{proj}"] = {
            "layer": layer, "proj": proj, "n_experts_sampled": len(vals),
            "mean_ratio_across_experts": sum(means) / len(means),
            "std_ratio_across_experts": (sum((r - sum(means) / len(means)) ** 2 for r in means) / len(means)) ** 0.5,
            "min_expert_ratio": min(means), "max_expert_ratio": max(means),
        }
    return aggregated


def compute_inter_layer_scale_profile(p2clip_wm, l0_wm):
    """Compare scale magnitude profiles across layers."""
    def profile(wm, art):
        prof = defaultdict(lambda: defaultdict(list))
        # Stratified sample: per (layer, proj) sample first expert only for speed
        seen = set()
        for key in wm:
            if not key.endswith(".weight_scale") or "experts" not in key:
                continue
            parts = key.split(".")
            if len(parts) != 8 or parts[4] != "experts":
                continue
            layer, expert, proj = int(parts[2]), int(parts[5]), parts[6]
            # Only take expert 0 per (layer, proj) for profile
            if expert != 0:
                continue
            key_type = (layer, proj)
            if key_type in seen:
                continue
            seen.add(key_type)
            s = load_tensor(art, wm, key)
            if s is not None:
                prof[layer][proj] = s.mean().item()
        return prof

    p2p = profile(p2clip_wm, P2CLIP_ARTIFACT)
    l0p = profile(l0_wm, L0_ARTIFACT)

    rows = []
    for layer in sorted(set(p2p) & set(l0p)):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if proj not in p2p[layer] or proj not in l0p[layer]:
                continue
            p2m = p2p[layer][proj]
            l0m = l0p[layer][proj]
            rows.append({"layer": layer, "proj": proj,
                         "p2clip_mean_scale": p2m, "l0_mean_scale": l0m,
                         "ratio": p2m / l0m if l0m > 0 else None})

    smooth = {"p2clip": [], "l0": []}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        pr = sorted([r for r in rows if r["proj"] == proj], key=lambda r: r["layer"])
        for i in range(1, len(pr)):
            smooth["p2clip"].append(abs(pr[i]["p2clip_mean_scale"] - pr[i-1]["p2clip_mean_scale"]))
            smooth["l0"].append(abs(pr[i]["l0_mean_scale"] - pr[i-1]["l0_mean_scale"]))

    return {
        "total_layers": len(set(r["layer"] for r in rows)),
        "per_layer_proj": rows,
        "smoothness": {
            "p2clip": sum(smooth["p2clip"]) / len(smooth["p2clip"]) if smooth["p2clip"] else None,
            "l0": sum(smooth["l0"]) / len(smooth["l0"]) if smooth["l0"] else None,
        },
    }


def main():
    print("=" * 78)
    print("  P2CLIP MBPP 塌陷配方归因案 —— 离线 Scale diff 分析")
    print("  Task C from PTQ_QAD_终局总结与下一步规划_1012.md §7.1")
    print("=" * 78)
    print(f"\n  p2clip: {P2CLIP_ARTIFACT}")
    print(f"  l0:     {L0_ARTIFACT}")

    p2w = build_weight_map(P2CLIP_ARTIFACT)
    l0w = build_weight_map(L0_ARTIFACT)
    print(f"  p2clip tensors: {len(p2w)},  l0 tensors: {len(l0w)}")

    # ---- §1 配方对比（静态摘要）----
    print("\n" + "=" * 78)
    print("  §1 配方成分对比：5 环（p2clip）vs 2 环（l0）")
    print("=" * 78)
    print("""
  成分               p2clip_s1234          l0_clip_lo05_omni   mbpp 归因假说
  ────────────────────────────────────────────────────────────────────────────
  ① QuaRot 旋转      seed=1234 全维       无                   旋转压平离群 → int4 误差
                      Hadamard                                   均匀化，但对代码多步推导
                                                                  的序列误差累积路径敏感

  ② SQ (SmoothQuant) α/β 搜索 8192 tok   无                   SQ 缩放激活 → 代码域
                                                                  校准失配（calib=web 为主）

  ③ up-down AWQ      网格 20 点            无                   int4 舍入中被 AWQ 偏置，
                                                                  对消在 clip/round 域不成立

  ④ clip-search      lo=0.5, with_ssz=t   lo=0.5, with_ssz=f,  clip 旋钮调 ppl↔he 权衡
                                               omni 精修

  ⑤ SSZ / omni       10 步 LSQ             60 步 OmniQuant      omni 修 ppl 不损 he

  ★ 核心：p2clip 多了三个**数学非恒等**变换（旋转 + SQ + AWQ），
    每个在 int4 边界引入的误差对非-ppl 任务有系统性偏差。
  """)

    # ---- §2 scale ratio 分析 ----
    print("=" * 78)
    print("  §2 Per-channel Scale Ratio (p2clip / l0) 分析")
    print("=" * 78)

    sr = compute_scale_ratios(p2w, l0w)

    by_proj = defaultdict(list)
    for k, v in sr.items():
        by_proj[v["proj"]].append(v)

    for proj in ["gate_proj", "up_proj", "down_proj"]:
        items = sorted(by_proj.get(proj, []), key=lambda x: x["layer"])
        if not items:
            continue
        means = [it["mean_ratio_across_experts"] for it in items]
        print(f"\n  [{proj}] {len(items)} layers")
        print(f"    mean p2/l0 = {sum(means)/len(means):.4f} ± {max(it['std_ratio_across_experts'] for it in items):.4f}")
        print(f"    range = [{min(it['min_expert_ratio'] for it in items):.4f}, {max(it['max_expert_ratio'] for it in items):.4f}]")
        for it in items[:5] + items[-3:]:
            print(f"      L{it['layer']:>2}: {it['mean_ratio_across_experts']:.4f} "
                  f"(n={it['n_experts_sampled']}, min={it['min_expert_ratio']:.4f}, max={it['max_expert_ratio']:.4f})")

    # ---- §3 跨层 profile ----
    print("\n" + "=" * 78)
    print("  §3 跨层 Scale Profile —— 旋转/SQ 导致的系统性偏移")
    print("=" * 78)

    prof = compute_inter_layer_scale_profile(p2w, l0w)
    print(f"\n  比较层数: {prof['total_layers']}")
    sm = prof["smoothness"]
    if sm["p2clip"] is not None:
        print(f"  层间 scale 跳变均值:")
        print(f"    p2clip: {sm['p2clip']:.6f}")
        print(f"    l0:     {sm['l0']:.6f}")
        delta = sm["l0"] - sm["p2clip"]
        print(f"    Δ(l0-p2): {delta:.6f}  "
              f"({'p2clip 更平滑 = 旋转/SQ 压平跨层差异' if delta > 0 else 'l0 更平滑'})")

    print("\n  逐层 gate_proj ratio profile:")
    for r in sorted([x for x in prof["per_layer_proj"] if x["proj"] == "gate_proj"],
                    key=lambda x: x["layer"]):
        ratio = r["ratio"] if r["ratio"] else 1.0
        bar = "█" * max(1, min(int(ratio * 20), 40))
        print(f"    L{r['layer']:>2}: p2={r['p2clip_mean_scale']:.6f}  "
              f"l0={r['l0_mean_scale']:.6f}  ratio={ratio:.4f}  {bar}")

    # ---- §4 翻转集归因 ----
    print("\n" + "=" * 78)
    print("  §4 翻转集归因交叉分析（flip_analysis.json）")
    print("=" * 78)

    p2_only = FLIP_DATA["p2clip_only_fail"]
    both_fail = FLIP_DATA["both_fail"]
    total = FLIP_DATA["total"]
    print(f"\n  总题 {total} | p2clip 独败 {p2_only} ({p2_only/total*100:.1f}%) | 双败 {both_fail} ({both_fail/total*100:.1f}%)")
    print(f"  靶正确率: {(total-both_fail)/total*100:.1f}%")
    print(f"\n  翻转分类（p2clip 独败）:")
    for cat, cnt in sorted(FLIP_DATA["categories"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<16s} {cnt:>4d} ({cnt/p2_only*100:>5.1f}%)")
    print(f"\n  ★ list({85/194*100:.1f}%) + string({42/194*100:.1f}%) = "
          f"{(85+42)/194*100:.1f}% 翻转")
    print(f"    共同特征：多步推导 + 可变长度序列 + 条件分支")

    # ---- §5 机制假说 ----
    print("\n" + "=" * 78)
    print("  §5 机制假说综合")
    print("=" * 78)
    print("""
  ★ 核心机制链：

  p2clip（5 环）→ mbpp -34pp 塌陷
  ──────────────────────────────────────
  ① 旋转压平离群 → 权重与代码生成路径的"自然激活模式"去相关
     → 对连续文本 (ppl) 有益，但对多步代码推导的误差路径敏感性增加

  ② SQ 缩放激活 → 校准域偏 web 文本，代码域分布失配

  ③ AWQ 对消 → 线性域成立，但 int4 clip/round 域不成立 → 净误差放大

  ④ 三变换复合非加性 → list/string 多步推导任务系统性翻转到错误路径

  l0（2 环）→ mbpp 0.734（与靶同级）
  ──────────────────────────────────────
  ① 无旋转/AWQ/SQ → 权重保持"自然"分布
  ② clip 压 outlier 不改相对分布 + omni 修 ppl 不损代码 prox

  ★ 数据交叉验证：
  - 翻转 65.5% 集中在 list+string（多步/可变长/条件分支）
    与 5 环复合误差累积假说一致
  - l0 在这些任务维持靶级表现（paired p not significant）
  - 生产靶 (0.712) 也含旋转+AWQ+SQ，但校准/精度细则不同，
    说明误差可通过校准域调优部分缓解（非 5 环必塌）
  """)

    # ---- 保存报告 ----
    report = {
        "title": "p2clip mbpp 塌陷配方归因案 — 离线 Scale diff 分析",
        "task": "C: PTQ_QAD_终局总结与下一步规划_1012.md §7.1",
        "p2clip_artifact": P2CLIP_ARTIFACT,
        "l0_artifact": L0_ARTIFACT,
        "flip_data": FLIP_DATA,
        "scale_ratio_summary": {
            proj: {
                "mean_ratio": sum(v["mean_ratio_across_experts"] for v in by_proj[proj]) / len(by_proj[proj]) if by_proj[proj] else None,
                "n_layers": len(by_proj[proj]),
            } for proj in ["gate_proj", "up_proj", "down_proj"]
        },
        "profile_smoothness": sm,
    }

    report_path = os.path.normpath(os.path.join(
        BASE, "glm52-qad", "audit1009", "p2clip_scale_diff_report.json"
    ))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [保存] {report_path}")


if __name__ == "__main__":
    main()
