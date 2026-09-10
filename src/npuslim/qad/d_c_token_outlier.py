"""D-c：per-token 离群度测量（0908 文档 §2.3.4 缺口 #2 的钉死实验）。

背景：Step A 测得 48× 离群是 **per-channel abs-mean 的跨通道 max/mean**（通道均值口径）；
而 per-token 动态量化的实际误差由 **token 内 absmax / 典型值** 决定——两个不同统计量。
persistent 通道级离群时二者一致，spiky 离群时前者高估后者。本实验逐层实测两个口径，
并同时给出 R1 旋转（x @ R1ᵀ，与量化/评测 hook 同方向）前后的对比——一次前向同时产出
raw 与 rotated 两套统计（同一批激活精确对比）。

统计量（逐层、逐 hook 点）：
- per-token ratio_max_mean = max_c|x_t| / mean_c|x_t|（token 内 absmax 与均值之比，
  直接对应 per-token 对称量化中"离群通道吃掉普通通道分辨率"的倍数）
- per-token ratio_max_rms  = max_c|x_t| / rms_c(x_t)
- 通道均值口径（Step A 对照）：max_c(mean_t|x|) / mean_c,t|x|（旋转前后同算）
- 高斯 iid 参考值：E[max/mean_abs] ≈ 4.4（dim=2048，供解读参考）

裁决（0908 §2.3.4 D-c）：
- 若 raw per-token ratio ≈ 通道均值口径（如 40×+）→ Step A 的 48× 有效传导到 per-token
  量化，旋转理论收益大；若 raw per-token ratio 远低于 48×（如 <10×）→ 通道均值口径高估，
  A7 "零收益"张力部分来自此缝隙；
- rotated per-token ratio 应显著回落（趋向 ~4-5 高斯基线）→ 旋转确实摊平 per-token 分布
  （机制有效），结合 A7 Step B 零收益即可锁定"生态位被 AWQ/动态 scale 占据"的归因。

运行（容器内，NPU 单卡）：
    ASCEND_RT_VISIBLE_DEVICES=4 python -u -m npuslim.qad.d_c_token_outlier \
        --num-samples 64 --max-seq-length 2048 --batch-size 4 --device npu \
        --output <path>/d_c_token_outlier.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from npuslim.qad.rotation import QuaRotMode, create_rot


# ---------------- 统计 hook ----------------

def make_stats_hook(acc: Dict, r1_t: torch.Tensor) -> callable:
    """累积 per-token 离群比（raw 与 rotated 同批计算）+ 通道 abs-mean（Step A 对照）。

    acc 结构（全部在 hook 注册设备上累积 fp32）：
      raw_{mm,rms}_sum / rot_{mm,rms}_sum : per-token ratio 的 sum（max/mean 与 max/rms）
      raw_cnt / rot_cnt                   : token 数
      ch_sum / ch_n                       : 通道 abs-sum（raw 激活，Step A 口径）
      ch_rot_sum                          : 通道 abs-sum（rotated 激活）
    """

    def hook(module, inp, out) -> None:
        x = out.float().reshape(-1, out.shape[-1])          # [N, H]
        with torch.no_grad():
            a = x.abs()
            # raw per-token ratios
            tmax = a.max(dim=-1).values
            tmean = a.mean(dim=-1)
            trms = x.pow(2).mean(dim=-1).sqrt()
            acc["raw_mm_sum"] += (tmax / tmean.clamp_min(1e-8)).sum()
            acc["raw_rms_sum"] += (tmax / trms.clamp_min(1e-8)).sum()
            acc["raw_cnt"] += x.shape[0]
            # rotated per-token ratios（同一批激活）
            xr = x @ r1_t
            ar = xr.abs()
            rmax = ar.max(dim=-1).values
            rmean = ar.mean(dim=-1)
            rrms = xr.pow(2).mean(dim=-1).sqrt()
            acc["rot_mm_sum"] += (rmax / rmean.clamp_min(1e-8)).sum()
            acc["rot_rms_sum"] += (rmax / rrms.clamp_min(1e-8)).sum()
            acc["rot_cnt"] += x.shape[0]
            # 通道 abs-mean（Step A 对照口径，raw 与 rotated）
            acc["ch_sum"] += a.sum(dim=0)
            acc["ch_rot_sum"] += ar.sum(dim=0)
            acc["ch_n"] += x.shape[0]

    return hook


def register_hooks(model, r1_t: torch.Tensor) -> List[tuple]:
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    handles: List[tuple] = []
    for i, layer in enumerate(layers):
        for path, mod in (("attn", layer.input_layernorm),
                          ("mlp", layer.post_attention_layernorm)):
            acc = {
                "raw_mm_sum": torch.zeros((), dtype=torch.float32, device=device),
                "raw_rms_sum": torch.zeros((), dtype=torch.float32, device=device),
                "rot_mm_sum": torch.zeros((), dtype=torch.float32, device=device),
                "rot_rms_sum": torch.zeros((), dtype=torch.float32, device=device),
                "raw_cnt": 0, "rot_cnt": 0,
                "ch_sum": torch.zeros(mod.weight.shape[0], dtype=torch.float32, device=device),
                "ch_rot_sum": torch.zeros(mod.weight.shape[0], dtype=torch.float32, device=device),
                "ch_n": 0,
            }
            h = mod.register_forward_hook(make_stats_hook(acc, r1_t))
            handles.append((i, path, h, acc))
    return handles


# ---------------- 采集与报告 ----------------

def collect(model, dataset, batch_size: int, device: torch.device, r1_t: torch.Tensor) -> Dict:
    handles = register_hooks(model, r1_t)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=type(dataset).collate_fn)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            model.model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            print(f"    calib step {step + 1}/{len(loader)} [{time.time() - t0:.0f}s]",
                  flush=True)
    stats: Dict[str, dict] = {}
    for i, path, h, acc in handles:
        h.remove()
        ch = acc["ch_sum"] / max(acc["ch_n"], 1)
        ch_rot = acc["ch_rot_sum"] / max(acc["ch_n"], 1)
        stats[f"{i}:{path}"] = {
            "tok_ratio_max_mean": float(acc["raw_mm_sum"] / max(acc["raw_cnt"], 1)),
            "tok_ratio_max_rms": float(acc["raw_rms_sum"] / max(acc["raw_cnt"], 1)),
            "tok_ratio_max_mean_rot": float(acc["rot_mm_sum"] / max(acc["rot_cnt"], 1)),
            "tok_ratio_max_rms_rot": float(acc["rot_rms_sum"] / max(acc["rot_cnt"], 1)),
            "chan_ratio": float(ch.max() / ch.mean().clamp_min(1e-8)),
            "chan_ratio_rot": float(ch_rot.max() / ch_rot.mean().clamp_min(1e-8)),
        }
    return stats


def build_report(stats: Dict[str, dict], meta: dict) -> dict:
    layers = []
    agg = {"raw_mm": [], "rot_mm": [], "raw_chan": [], "rot_chan": [],
           "raw_rms": [], "rot_rms": []}
    for key in sorted(stats.keys(), key=lambda k: int(k.split(":")[0])):
        i, path = key.split(":")
        s = stats[key]
        layers.append({"layer": int(i), "path": path, **{k: round(v, 3) for k, v in s.items()}})
        agg["raw_mm"].append(s["tok_ratio_max_mean"])
        agg["rot_mm"].append(s["tok_ratio_max_mean_rot"])
        agg["raw_rms"].append(s["tok_ratio_max_rms"])
        agg["rot_rms"].append(s["tok_ratio_max_rms_rot"])
        agg["raw_chan"].append(s["chan_ratio"])
        agg["rot_chan"].append(s["chan_ratio_rot"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def maxv(xs):
        return max(xs) if xs else 0.0

    summary = {
        "per_token_max_mean_mean": round(mean(agg["raw_mm"]), 3),
        "per_token_max_mean_max": round(maxv(agg["raw_mm"]), 3),
        "per_token_max_mean_rot_mean": round(mean(agg["rot_mm"]), 3),
        "per_token_max_rms_mean": round(mean(agg["raw_rms"]), 3),
        "per_token_max_rms_rot_mean": round(mean(agg["rot_rms"]), 3),
        "channel_mean_ratio_mean": round(mean(agg["raw_chan"]), 3),
        "channel_mean_ratio_max": round(maxv(agg["raw_chan"]), 3),
        "channel_mean_ratio_rot_mean": round(mean(agg["rot_chan"]), 3),
        "gauss_iid_ref_max_mean": 4.4,
        "step_a_channel_ratio_ref": 48.4,
    }
    return {"meta": meta, "summary": summary, "layers": layers}


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="D-c: per-token 离群度测量（旋转前后同批对比）")
    parser.add_argument("--model-path", default="/llm_infer_workspace/lichc/models/"
                        "Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--calib-path", default="dataset/calib/mixed_chat_code.jsonl")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--r1-seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device in ("npu", "npu:0"):
        device_map = {"": "npu:0"}
    elif args.device == "cuda":
        device_map = {"": "cuda:0"}
    else:
        device_map = args.device

    t0 = time.time()
    print(f"[d_c] 加载模型 {args.model_path} (device_map={device_map}) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device_map, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = model.model.embed_tokens.weight.device
    print(f"[d_c] 模型加载完成 {time.time() - t0:.0f}s, device={device}", flush=True)

    hidden = model.config.hidden_size
    r1 = create_rot(QuaRotMode.HADAMARD, hidden, block_size=args.block_size,
                    seed=args.r1_seed).float().to(device)
    r1_t = r1.T.contiguous()
    print(f"[d_c] R1 {tuple(r1.shape)} seed={args.r1_seed}（与 A7 量化/评测同参）", flush=True)

    from npuslim.datasets.mixed_domain_dataset import MixedDomainDataset
    calib = MixedDomainDataset(
        data_path=args.calib_path,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        processor=tokenizer,
        device="cpu",
        seed=args.seed,
    )
    print(f"[d_c] 校准数据 {len(calib)} 样本 x {args.max_seq_length} tokens", flush=True)

    stats = collect(model, calib, args.batch_size, device, r1_t)
    meta = {
        "model": args.model_path, "calib": args.calib_path,
        "num_samples": len(calib), "max_seq_length": args.max_seq_length,
        "r1_seed": args.r1_seed, "block_size": args.block_size,
        "device": str(device),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = build_report(stats, meta)

    s = report["summary"]
    print("\n" + "=" * 96)
    print(f"{'口径':<34}{'raw':>10}{'rotated':>10}{'参考':>12}")
    print("-" * 96)
    print(f"{'per-token max/mean_abs（逐层均值）':<34}{s['per_token_max_mean_mean']:>10}"
          f"{s['per_token_max_mean_rot_mean']:>10}{'~4.4':>12}")
    print(f"{'per-token max/rms（逐层均值）':<34}{s['per_token_max_rms_mean']:>10}"
          f"{s['per_token_max_rms_rot_mean']:>10}{'':>12}")
    print(f"{'channel-mean max/mean（Step A 口径，逐层均值）':<34}{s['channel_mean_ratio_mean']:>10}"
          f"{s['channel_mean_ratio_rot_mean']:>10}{'48.4(StepA)':>12}")
    print(f"{'channel-mean max/mean（全模型最大）':<34}{s['channel_mean_ratio_max']:>10}"
          f"{'':>10}{'48.4(StepA)':>12}")
    print("=" * 96)

    out_path = args.output or "d_c_token_outlier.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[d_c] 报告已写入 {out_path}")


if __name__ == "__main__":
    main()
