#!/usr/bin/env python3
"""flip_trajectory_filter.py — 方案 A：用 ckpt 轨迹过滤翻转方向。

核心思路：比较 step600 和 step1400 的每元素翻转方向，只保留被"确认"的翻转。
- step600 翻转了且 step1400 保持同方向翻转 = 确认 → 保留
- step600 翻转了但 step1400 被训练推回了 = 噪音 → 过滤掉（用原码）
- 从来不翻转 = 直通

输出：一个"去噪交付物"（W4A8 artifact，同 reencode 输出格式）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LO_CLIP = "/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni"
CKPT_DIR = "/llm_infer_workspace/ylt/glm52-qad/ckpt_e2"
SCALING = 4.0
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
COMPANIONS = (".weight_scale", ".weight_offset", ".scale_bias")


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    u8 = (p.to(torch.int32) & 0xFF).to(torch.uint8)
    lo = (u8 & 0x0F).to(torch.int32)
    hi = ((u8 >> 4) & 0x0F).to(torch.int32)
    lo = lo - 16 * (lo >= 8).to(torch.int32)
    hi = hi - 16 * (hi >= 8).to(torch.int32)
    out = torch.empty(p.shape[0] * 2, p.shape[1], dtype=torch.int32)
    out[0::2], out[1::2] = lo, hi
    return out


def pack_int4(signed: torch.Tensor) -> torch.Tensor:
    x = signed.to(torch.int32) & 0x0F
    lo = x[0::2]
    hi = x[1::2]
    return (lo | (hi << 4)).to(torch.int8)


def delta_for(sd: dict, layer: int, expert: int, proj: str,
              scaling: float = SCALING) -> torch.Tensor | None:
    p = f"model.layers.{layer}.mlp.experts.lora.params"
    try:
        if proj in ("gate_proj", "up_proj"):
            bgu = sd[f"{p}.Bgu"][expert].float()
            agu = sd[f"{p}.Agu"][expert].float()
            d = (bgu @ agu) * scaling
            half = d.shape[0] // 2
            return d[:half] if proj == "gate_proj" else d[half:]
        bdn = sd[f"{p}.Bdn"][expert].float()
        adn = sd[f"{p}.Adn"][expert].float()
        return (bdn @ adn) * scaling
    except (KeyError, IndexError):
        return None


def compute_q_new(q_base: torch.Tensor, d: torch.Tensor,
                  s: torch.Tensor) -> torch.Tensor:
    """Compute the new quantized code: clamp(round((deq + d) / s), -8, 7)"""
    q_signed = unpack_int4(q_base).float()
    deq = q_signed * s
    wp = deq + d
    return torch.clamp(torch.round(wp / s), -8, 7).to(torch.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=LO_CLIP)
    ap.add_argument("--lora-ckpt", required=True,
                    help="ckpt to use as the base (e.g. step600)")
    ap.add_argument("--direction-ckpt", required=True,
                    help="ckpt to use for direction validation (e.g. step1400)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", type=float, default=SCALING)
    ap.add_argument("--shard-gb", type=float, default=8.0)
    args = ap.parse_args()

    ts0 = time.time()
    print(f"[flip_filter] 方案 A: 翻转轨迹过滤", flush=True)
    print(f"[flip_filter] 底座: {args.artifact}", flush=True)
    print(f"[flip_filter] 主 ckpt: {args.lora_ckpt}", flush=True)
    print(f"[flip_filter] 方向 ckpt: {args.direction_ckpt}", flush=True)
    print(f"[flip_filter] 输出: {args.out}", flush=True)
    print("=" * 60, flush=True)

    # 加载底座 index
    with open(f"{args.artifact}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    print(f"[flip_filter] 底座张量: {len(wm)}", flush=True)

    # 加载两份 ckpt
    print(f"[flip_filter] 加载 {args.lora_ckpt} ...", flush=True)
    sd_main = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    print(f"[flip_filter] 加载 {args.direction_ckpt} ...", flush=True)
    sd_dir = torch.load(args.direction_ckpt, map_location="cpu", weights_only=False)
    print(f"[flip_filter] ckpt 加载完成 ({time.time()-ts0:.1f}s)", flush=True)

    # 创建输出目录，拷贝非 safetensors 配置文件
    os.makedirs(args.out, exist_ok=True)
    for f in sorted(os.listdir(args.artifact)):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        if os.path.isfile(f"{args.artifact}/{f}"):
            shutil.copy2(f"{args.artifact}/{f}", f"{args.out}/{f}")

    # 逐 shard 处理
    shard_tensors: dict[str, list[str]] = {}
    for name, shard in wm.items():
        shard_tensors.setdefault(shard, []).append(name)

    max_bytes = int(args.shard_gb * 2**30)
    index: dict[str, str] = {}
    total_size = 0
    shard_id = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0

    stats = {"reencode_expert": 0, "passthrough_w8a8": 0, "passthrough_float": 0,
             "companion": 0}
    total_flip_main = 0      # step600 翻转数
    total_flip_filtered = 0  # 过滤后保留的翻转数
    n_confirmed = 0           # 翻转方向一致的
    n_reverted = 0            # 方向相反/回退的
    n_new_dir = 0             # 向新方向翻的（不管它，直通）

    def flush_shard():
        nonlocal shard_id, buf, buf_bytes, total_size
        if not buf:
            return
        fname = f"qad_flip_filtered-{shard_id:05d}.safetensors"
        save_file(buf, f"{args.out}/{fname}", metadata={"format": "pt"})
        for k in buf:
            index[k] = fname
        total_size += sum(t.untyped_storage().nbytes() for t in buf.values())
        print(f"[flip_filter] shard {fname}: {len(buf)} tensors, "
              f"{buf_bytes / 2**30:.2f} GiB ({time.time()-ts0:.0f}s)", flush=True)
        shard_id += 1
        buf, buf_bytes = {}, 0

    for shard_file in sorted(shard_tensors):
        with safe_open(f"{args.artifact}/{shard_file}", framework="pt",
                       device="cpu") as fd:
            for name in shard_tensors[shard_file]:
                base = name[:-7] if name.endswith(".weight") else None

                # companion 直通
                if name.endswith(COMPANIONS):
                    buf[name] = fd.get_tensor(name).contiguous()
                    buf_bytes += buf[name].untyped_storage().nbytes()
                    stats["companion"] += 1
                    if buf_bytes >= max_bytes:
                        flush_shard()
                    continue

                if base is not None and f"{base}.weight_scale" in wm:
                    q_base = fd.get_tensor(name)
                    s = load_tensor(wm, args.artifact, f"{base}.weight_scale",
                                    device="cpu").float().reshape(-1, 1)

                    if q_base.dtype == torch.int8 and q_base.shape[0] * 2 == s.shape[0]:
                        # 专家 int4
                        m = EXPERT_RE.match(name)
                        if not m:
                            buf[name] = q_base.contiguous()
                            buf_bytes += buf[name].untyped_storage().nbytes()
                            stats["passthrough_w8a8"] += 1
                            if buf_bytes >= max_bytes:
                                flush_shard()
                            continue
                        layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)

                        # step600 的 δ
                        d_main = delta_for(sd_main, layer, expert, proj)
                        # step1400 的 δ
                        d_dir = delta_for(sd_dir, layer, expert, proj)

                        if d_main is None or d_dir is None:
                            # 无 LoRA → 直通
                            buf[name] = q_base.contiguous()
                            buf_bytes += buf[name].untyped_storage().nbytes()
                            stats["passthrough_w8a8"] += 1
                            if buf_bytes >= max_bytes:
                                flush_shard()
                            continue

                        # 计算 step600 的码
                        q_main = compute_q_new(q_base, d_main, s)
                        # 计算 step1400 的码
                        q_dir = compute_q_new(q_base, d_dir, s)
                        # 底座原码
                        q_old = unpack_int4(q_base)

                        # 翻转判断
                        flip_main = (q_main != q_old)    # 哪些元素在 step600 翻转了
                        n_flip_main = int(flip_main.sum().item())
                        total_flip_main += n_flip_main

                        # 方向一致性：在 step600 翻转的元素，step1400 是否保持同向
                        # 同向 = q_main[i] 到 q_dir[i] 同号 or q_dir[i] == q_main[i] (保持)
                        # 反向 = q_dir[i] → q_old (回退) 或 q_dir[i] 到另一方向
                        # 准则：用 q_dir 作为"正确方向"的参考
                        # 如果 q_dir == q_old → 训练完全推回了 → 用 q_old（过滤）
                        # 如果 q_dir != q_old 但 q_dir != q_main → 方向漂移 → 保守起见用 q_dir
                        # 如果 q_dir == q_main → 确认 → 用 q_main

                        # 默认用 step600 的码（即 q_main）
                        q_final = q_main.clone()

                        # 对于 step600 翻转了的元素：
                        mask = flip_main
                        # subset: 只关心 flip_main 为 True 的位置
                        idx = mask.nonzero(as_tuple=True)
                        # 对每个翻转元素做决策
                        for i in range(idx[0].shape[0]):
                            row, col = idx[0][i].item(), idx[1][i].item()
                            qm = q_main[row, col].item()
                            qd = q_dir[row, col].item()
                            qo = q_old[row, col].item()

                            if qm == qd:
                                # 确认：step600 和 step1400 向同一方向翻
                                n_confirmed += 1
                                q_final[row, col] = qm
                            elif qd == qo:
                                # 回退：step1400 回到了原码 → 训练后悔了
                                n_reverted += 1
                                q_final[row, col] = qo
                            else:
                                # 漂移：step1400 翻到了另一个方向
                                # 使用 step1400 的方向（更新的训练信号）
                                n_reverted += 1
                                q_final[row, col] = qd

                        # 统计过滤后与 q_old 不同的数量
                        n_flip_filtered = int((q_final != q_old).sum().item())
                        total_flip_filtered += n_flip_filtered

                        buf[name] = pack_int4(q_final).contiguous()
                        buf_bytes += buf[name].untyped_storage().nbytes()
                        stats["reencode_expert"] += 1
                    else:
                        # W8A8: 直通
                        buf[name] = q_base.contiguous()
                        buf_bytes += buf[name].untyped_storage().nbytes()
                        stats["passthrough_w8a8"] += 1
                    if buf_bytes >= max_bytes:
                        flush_shard()
                    continue

                # float 直通
                buf[name] = fd.get_tensor(name).contiguous()
                buf_bytes += buf[name].untyped_storage().nbytes()
                stats["passthrough_float"] += 1
                if buf_bytes >= max_bytes:
                    flush_shard()
    flush_shard()

    # 写入 index
    with open(f"{args.out}/model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": index}, f, indent=2)

    print(f"\n[flip_filter] {'='*60}", flush=True)
    print(f"[flip_filter] DONE ({time.time()-ts0:.0f}s)", flush=True)
    print(f"[flip_filter] 张量数: {len(index)}", flush=True)
    print(f"[flip_filter] total_size: {total_size / 2**30:.1f} GiB", flush=True)
    print(f"[flip_filter] stats: {stats}", flush=True)
    print(f"[flip_filter] step600 翻转: {total_flip_main:,}", flush=True)
    print(f"[flip_filter] 过滤后保留翻转: {total_flip_filtered:,}", flush=True)
    print(f"[flip_filter] 确认(同向): {n_confirmed:,}", flush=True)
    print(f"[flip_filter] 回退/漂移: {n_reverted:,}", flush=True)
    print(f"[flip_filter] 过滤比例: {(n_reverted/(n_confirmed+n_reverted))*100:.1f}%"
          if (n_confirmed+n_reverted) > 0 else "", flush=True)
    print(f"[flip_filter] {'='*60}", flush=True)


def load_tensor(wm: dict, art: str, name: str, device="cpu") -> torch.Tensor:
    with safe_open(f"{art}/{wm[name]}", framework="pt", device=device) as f:
        return f.get_tensor(name)


if __name__ == "__main__":
    main()
