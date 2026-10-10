"""A7 Step A：真实 LLM 残差流 per-channel abs-mean 离群度测量（调研文档 §6.1 下一步）。

目标：为 "R1（残差流→qkv/router/gate_up 输入）是否值得立项 CANN kernel" 提供
go/no-go 前置信号——在真实 LLM 校准前向上测量残差流入口（norm 后、量化 matmul
边界）的 per-channel abs-mean 离群度（max/mean），≥10× 则 Step B 排入（§6.1 /
§6.2 D2）。

为什么 Qwen3-30B-A3B 先行（2026-09-04 裁定）：
- BF16 57GB，单卡（Ascend910 64GB / 双 RTX5090）即可全量加载，**非流式**，
  与 GLM-5.2（1.4TB，需流式）相比实现与迭代成本低一个量级；
- 标准 MoE 架构，无 DSA indexer / MLA 干扰，hook 点干净；
- 与 M4 QAD 同模型（Qwen3-30B 自蒸馏），A7 结论可直接对接；
- W4A8 基线现成（§12.7：humaneval -4.27pp / wikitext +9.6%）。

hook 点（与 tiny A7 的 R1 注入点一致 = norm 后、量化 matmul 边界，
modeling_qwen3_moe.py 每层）：
- layer.input_layernorm 输出       -> self_attn（q/k/v/o_proj）
- layer.post_attention_layernorm 输出 -> mlp（router gate + experts gate_up）
两者都在残差流空间（dim=hidden_size），R1 用同一旋转矩阵。

产物仅用于离群度统计（Step A 报告 json），不产出任何量化产物（不可部署）。

运行示例（本地 GPU）：
    CUDA_VISIBLE_DEVICES=0,1 /data/yult/miniconda3/envs/npuslim/bin/python \
        -m npuslim.qad.step_a_outlier \
        --num-samples 32 --max-seq-length 1024 --batch-size 4
"""
from __future__ import annotations

# ===== Phase D 路径参数化（env 带默认回退）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from npuslim.core import ModelRegistry
from npuslim.datasets.mixed_domain_dataset import MixedDomainDataset


# ---------------- 统计原语 ----------------

def outlier_ratio(per_channel_abs_mean: torch.Tensor) -> float:
    """per-channel abs-mean 的 max/mean（>>1 说明存在离群通道）。

    与 tiny A7 harness 的 outlier_ratio 同语义（tests/qad/test_a7_rotation.py）。
    """
    cm = per_channel_abs_mean.float()
    return (cm.max() / cm.mean().clamp_min(1e-8)).item()


def make_hook(acc: Dict[str, torch.Tensor]) -> callable:
    """构造累积 per-channel abs-sum 的 forward hook（out: [B, L, H]）。

    与 tiny 统计语义一致：x_abs_mean = Σ|x| / n（逐通道）。hook 输出为 bf16，
    先转 float32 再累积，避免精度漂移。
    """

    def hook(module, inp, out) -> None:
        x = out.float().reshape(-1, out.shape[-1])
        acc["sum"].add_(x.abs().sum(dim=0))
        acc["n"] += x.shape[0]

    return hook


# ---------------- 采集 ----------------

def register_residual_hooks(model) -> List[tuple]:
    """在每层 input_layernorm / post_attention_layernorm 输出注册 hook。

    返回 (layer_idx, path, hook_handle, acc) 列表，调用方负责 remove。
    """
    layers = model.model.layers  # Qwen3MoeModel.layers
    device = model.model.embed_tokens.weight.device  # 与权重同设备（NPU/CUDA）
    handles: List[tuple] = []
    for i, layer in enumerate(layers):
        for path, mod in (("attn", layer.input_layernorm),
                          ("mlp", layer.post_attention_layernorm)):
            acc = {"sum": torch.zeros(mod.weight.shape[0], dtype=torch.float32,
                                      device=device),
                   "n": 0}
            h = mod.register_forward_hook(make_hook(acc))
            handles.append((i, path, h, acc))
    return handles


def collect_stats(model, dataset: MixedDomainDataset, batch_size: int,
                  device: torch.device) -> Dict[str, torch.Tensor]:
    """跑一遍校准前向（model.model 主干，跳过 lm_head），累积残差流统计。"""
    handles = register_residual_hooks(model)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=MixedDomainDataset.collate_fn)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model.model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"    calib step {step + 1}/{len(loader)}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    for _, _, h, _ in handles:
        h.remove()
    # 组装 per-layer 结构：{(layer, path): abs_mean}（转 CPU，报告计算与设备无关）
    stats: Dict[str, torch.Tensor] = {}
    for i, path, _, acc in handles:
        stats[f"{i}:{path}"] = (acc["sum"].float() / max(acc["n"], 1)).cpu()
    return stats


# ---------------- 报告 ----------------

def build_report(stats: Dict[str, torch.Tensor], meta: dict) -> dict:
    layers: List[dict] = []
    summary = {"attn_max_ratio": 0.0, "mlp_max_ratio": 0.0,
               "attn_layers_ge10": 0, "mlp_layers_ge10": 0}
    for key in sorted(stats.keys(), key=lambda k: int(k.split(":")[0])):
        layer_idx, path = key.split(":")
        cm = stats[key]
        ratio = outlier_ratio(cm)
        max_v, max_i = cm.max(dim=0)
        top = torch.topk(cm, k=min(10, cm.numel()))
        entry = {
            "layer": int(layer_idx),
            "path": path,
            "max": float(max_v),
            "mean": float(cm.mean()),
            "ratio": ratio,
            "top_channels": [int(i) for i in top.indices.tolist()],
            "top_vals": [float(v) for v in top.values.tolist()],
        }
        layers.append(entry)
        tag = "attn" if path == "attn" else "mlp"
        summary[f"{tag}_max_ratio"] = max(summary[f"{tag}_max_ratio"], ratio)
        if ratio >= 10.0:
            summary[f"{tag}_layers_ge10"] += 1
    verdict = ("STEP B: 达 ≥10× 门槛，R1 CANN kernel 值得立项评估（§6.2 D2 放行）"
               if (summary["attn_max_ratio"] >= 10.0 or summary["mlp_max_ratio"] >= 10.0)
               else "未达 ≥10× 门槛，R1 收益证据不足，归档数据、资源转 QAD")
    return {"meta": meta, "layers": layers, "summary": summary, "verdict": verdict}


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A7 Step A: 残差流离群度测量")
    parser.add_argument("--model-path", default=f"{SHARED}/lichc/models/"
                        "Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--model-hub", default="ms")
    parser.add_argument("--calib-path", default="dataset/calib/mixed_chat_code.jsonl")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto",
                        help="device_map for full model load: auto/cpu/cuda/npu")
    parser.add_argument("--output", default=None, help="report json 输出路径")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # device_map 归一化：npu/cuda 单设备全量放置用 {"": "npu:0"}/{"": "cuda:0"}，
    # 避免 accelerate 对裸 "npu" 字符串解析失败；auto/cpu 原样透传。
    if args.device in ("npu", "npu:0"):
        device_map: dict = {"": "npu:0"}
    elif args.device == "cuda":
        device_map = {"": "cuda:0"}
    else:
        device_map = args.device  # "auto" / "cpu" / "cuda:0" / ...
    t0 = time.time()
    print(f"[step_a] 加载模型 {args.model_path} (device_map={device_map}) ...", flush=True)
    wrapper = ModelRegistry.create("Qwen3", path=args.model_path,
                                   model_hub=args.model_hub,
                                   device_map=device_map,
                                   torch_dtype=torch.bfloat16,
                                   trust_remote_code=True,
                                   low_cpu_mem_usage=True)
    wrapper.prepare_full_model()
    model = wrapper.model
    device = model.model.embed_tokens.weight.device
    print(f"[step_a] 模型加载完成 {time.time() - t0:.0f}s，前向设备 {device}", flush=True)

    calib = MixedDomainDataset(
        data_path=args.calib_path,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        processor=wrapper.tokenizer,
        device="cpu",
        seed=args.seed,
    )
    print(f"[step_a] 校准数据 {len(calib)} 样本 x {args.max_seq_length} tokens", flush=True)

    stats = collect_stats(model, calib, args.batch_size, device)
    meta = {
        "model": args.model_path,
        "model_type": "qwen3_moe",
        "calib": args.calib_path,
        "num_samples": args.num_samples,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = build_report(stats, meta)

    # 控制台逐层表
    print("\n" + "=" * 88)
    print(f"{'layer':>5} {'path':>4} {'max':>12} {'mean':>12} {'ratio':>8}  top-outlier ch")
    print("-" * 88)
    for e in report["layers"]:
        print(f"{e['layer']:>5} {e['path']:>4} {e['max']:>12.4f} {e['mean']:>12.4f} "
              f"{e['ratio']:>8.2f}  {e['top_channels'][:5]}")
    s = report["summary"]
    print("=" * 88)
    print(f"attn 路径最大离群比 = {s['attn_max_ratio']:.2f}（≥10× 层数 {s['attn_layers_ge10']}）")
    print(f"mlp  路径最大离群比 = {s['mlp_max_ratio']:.2f}（≥10× 层数 {s['mlp_layers_ge10']}）")
    print(f"判定: {report['verdict']}")

    out_path = args.output or "step_a_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[step_a] 报告已写入 {out_path}")


if __name__ == "__main__":
    main()
