#!/usr/bin/env python
"""reencode_qad_deliverable.py — QAD 交付 artifact 一步式 merge+re-encode（fp32 单 pass）。

语义（deploy-compose，与训练 fakequant 逐 bit 对齐——同 fp32 域、同网格、同 scale 继承）：
  专家权重（int4 打包）：deq = unpack_int4(q)×s + o（fp32，与训练 stream-load 同式）
                        w' = deq + (B@A)×scaling
                        q' = pack_int4(clamp(round(w'/s), -8, 7))
  W8A8（attention）：δ=0（QAD LoRA 仅覆盖专家）→ 原码直通（q/s/o 原样）
  companion（weight_scale/weight_offset/scale_bias）：原样直通（网格定义继承）
  float 直通（embeddings/norms 等）：原样直通
输出：与输入 artifact 同格式的交付 artifact（vLLM W4A8 直接可部署）。

无损性（deploy-compose 设计收益）：训练有效权重 = fakequant(W+δ) 已在部署网格上
（scale 继承自 artifact）→ 再编码零亚网格丢失（STEP4 post 范式的 70% 亚网格丢失
结构性消失）。注意：不用 bf16 中间目录（merge_qad_model.py 的 bf16 输出会引入
一层 bf16 预舍入，可能翻转边界码——本脚本 fp32 单 pass 逐 bit 对齐训练）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
COMPANIONS = (".weight_scale", ".weight_offset", ".scale_bias")


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    """W4A8 row-wise int4: [out//2, in] int8 -> signed int4 [out, in]."""
    u8 = (p.to(torch.int32) & 0xFF).to(torch.uint8)
    lo = (u8 & 0x0F).to(torch.int32)
    hi = ((u8 >> 4) & 0x0F).to(torch.int32)
    lo = lo - 16 * (lo >= 8).to(torch.int32)
    hi = hi - 16 * (hi >= 8).to(torch.int32)
    out = torch.empty(p.shape[0] * 2, p.shape[1], dtype=torch.int32)
    out[0::2] = lo
    out[1::2] = hi
    return out


def pack_int4(signed: torch.Tensor) -> torch.Tensor:
    """[out, in] signed int4 (-8..7) -> [out//2, in] int8（unpack 的精确逆）。

    unpack: out[2k]=lo(byte), out[2k+1]=hi(byte)
    pack:   byte k = (out[2k]&0xF) | ((out[2k+1]&0xF)<<4)
    往返自检：signed=[-1,2] → packed=0x2F → unpack → [-1,2] ✓"""
    x = signed.to(torch.int32) & 0x0F          # 补码 nibble（-8→8, -1→15, 7→7）
    lo = x[0::2]                               # 偶数行 → 低 nibble
    hi = x[1::2]                               # 奇数行 → 高 nibble
    return (lo | (hi << 4)).to(torch.int8)     # [out//2, in]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", type=float, default=4.0)   # alpha/rank = 64/16
    ap.add_argument("--shard-gb", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.artifact}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    print(f"[reenc] loading LoRA ckpt: {args.lora}", flush=True)
    sd = torch.load(args.lora, map_location="cpu", weights_only=False)

    shard_tensors: dict[str, list[str]] = {}
    for name, shard in wm.items():
        shard_tensors.setdefault(shard, []).append(name)
    print(f"[reenc] artifact tensors: {len(wm)} in {len(shard_tensors)} shards", flush=True)

    def load(name: str) -> torch.Tensor:
        with safe_open(f"{args.artifact}/{wm[name]}", framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def delta_for(layer: int, expert: int, proj: str) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp.experts.lora.params"
        if proj in ("gate_proj", "up_proj"):
            dgu = (sd[f"{p}.Bgu"][expert].float() @ sd[f"{p}.Agu"][expert].float()) * args.scaling
            half = dgu.shape[0] // 2
            return dgu[:half] if proj == "gate_proj" else dgu[half:]
        return (sd[f"{p}.Bdn"][expert].float() @ sd[f"{p}.Adn"][expert].float()) * args.scaling

    # 非 safetensors 文件（config/tokenizer/...）直通
    for f in sorted(os.listdir(args.artifact)):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        if os.path.isfile(f"{args.artifact}/{f}"):
            shutil.copy2(f"{args.artifact}/{f}", f"{args.out}/{f}")
            print(f"[reenc] copied {f}", flush=True)

    max_bytes = int(args.shard_gb * 2**30)
    index: dict[str, str] = {}
    total_size = 0
    shard_id = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    stats = {"reencode_expert": 0, "passthrough_w8a8": 0, "passthrough_float": 0,
             "companion": 0}
    d_samples: list[float] = []
    grid_flip = 0

    def flush_shard() -> None:
        nonlocal shard_id, buf, buf_bytes, total_size
        if not buf:
            return
        fname = f"qad600-{shard_id:05d}.safetensors"
        save_file(buf, f"{args.out}/{fname}", metadata={"format": "pt"})
        for k in buf:
            index[k] = fname
        total_size += sum(t.untyped_storage().nbytes() for t in buf.values())
        print(f"[reenc] shard {fname}: {len(buf)} tensors, "
              f"{buf_bytes / 2**30:.2f} GiB", flush=True)
        shard_id += 1
        buf, buf_bytes = {}, 0

    for shard_file in sorted(shard_tensors):
        with safe_open(f"{args.artifact}/{shard_file}", framework="pt",
                       device="cpu") as f:
            for name in shard_tensors[shard_file]:
                base = name[:-7] if name.endswith(".weight") else None
                # companion（scale/offset/scale_bias）：原样直通（网格定义继承）
                if name.endswith(COMPANIONS):
                    buf[name] = f.get_tensor(name).contiguous()
                    buf_bytes += buf[name].untyped_storage().nbytes()
                    stats["companion"] += 1
                    if buf_bytes >= max_bytes:
                        flush_shard()
                    continue
                if base is not None and f"{base}.weight_scale" in wm:
                    q = f.get_tensor(name)
                    s = load(f"{base}.weight_scale").float().reshape(-1, 1)
                    if q.dtype == torch.int8 and q.shape[0] * 2 == s.shape[0]:
                        # 专家 int4：fp32 反量化 + δ → 网格再编码（与训练 fakequant 逐 bit 对齐）
                        deq = (unpack_int4(q).float() * s
                               + (load(f"{base}.weight_offset").float().reshape(-1, 1)
                                  if f"{base}.weight_offset" in wm else torch.zeros(1)))
                        m = EXPERT_RE.match(name)
                        assert m, name
                        layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                        d = delta_for(layer, expert, proj)
                        if len(d_samples) < 40:
                            d_samples.append(d.abs().mean().item())
                        wp = deq + d
                        # 网格再编码：与训练 _STEQuant.forward 同式（fp32 div/round/clamp）
                        q_signed = torch.clamp(torch.round(wp / s), -8, 7)
                        # 网格翻转统计（诊断：δ 改变码位的比例）
                        q_old = unpack_int4(q)
                        grid_flip += int((q_signed != q_old).sum().item())
                        buf[name] = pack_int4(q_signed).contiguous()
                        buf_bytes += buf[name].untyped_storage().nbytes()
                        stats["reencode_expert"] += 1
                    else:
                        # W8A8（attention）：δ=0 → 原码直通
                        buf[name] = q.contiguous()
                        buf_bytes += buf[name].untyped_storage().nbytes()
                        stats["passthrough_w8a8"] += 1
                    if buf_bytes >= max_bytes:
                        flush_shard()
                    continue
                # float 直通（embeddings/norms 等）
                buf[name] = f.get_tensor(name).contiguous()
                buf_bytes += buf[name].untyped_storage().nbytes()
                stats["passthrough_float"] += 1
                if buf_bytes >= max_bytes:
                    flush_shard()
    flush_shard()

    with open(f"{args.out}/model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": index}, f, indent=2)

    print(f"[reenc] DONE: {len(index)} tensors, total_size={total_size / 2**30:.1f} GiB, "
          f"stats={stats}", flush=True)
    print(f"[reenc] 网格翻转（δ 改变码位的权重元素数）: {grid_flip}", flush=True)
    if d_samples:
        d_samples.sort()
        print(f"[reenc] delta |mean| samples: min={d_samples[0]:.3e} "
              f"median={d_samples[len(d_samples)//2]:.3e} max={d_samples[-1]:.3e} "
              f"(n={len(d_samples)})", flush=True)


if __name__ == "__main__":
    main()
