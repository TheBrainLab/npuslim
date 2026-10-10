#!/usr/bin/env python
"""Merge the QAD LoRA into the P1 base artifact -> merged bf16 model dir.

Semantics (Qwen3-30B M5-1 precedent): merged = artifact_dequant + delta, where
  artifact_dequant = unpack_int4(q)*scale + offset   (expert W4A8, offset==0)
                   = q*scale + offset                (attention/dense W8A8)
  delta = (B@A) * scaling  (QAD ckpt; fused gate_up first half = gate_proj).
Float passthrough tensors are copied as-is; companion tensors
(weight_scale/weight_offset/scale_bias) are consumed by the dequant and dropped.
The requant pipeline re-applies quantization on this merged model.

Output: a standard HF model dir (bf16 safetensors shards + index + copied
config/tokenizer files) that the GPTQ pipeline can consume directly.
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

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", type=float, default=4.0)
    ap.add_argument("--shard-gb", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.artifact}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    print(f"[merge] loading LoRA ckpt: {args.lora}", flush=True)
    sd = torch.load(args.lora, map_location="cpu", weights_only=False)

    shard_tensors: dict[str, list[str]] = {}
    for name, shard in wm.items():
        shard_tensors.setdefault(shard, []).append(name)
    print(f"[merge] artifact tensors: {len(wm)} in {len(shard_tensors)} shards", flush=True)

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

    # copy non-weight files (config/tokenizer/...)
    for f in sorted(os.listdir(args.artifact)):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        shutil.copy2(f"{args.artifact}/{f}", f"{args.out}/{f}")
        print(f"[merge] copied {f}", flush=True)

    max_bytes = int(args.shard_gb * 2**30)
    index: dict[str, str] = {}
    total_size = 0
    shard_id = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    stats = {"dequant_expert": 0, "dequant_other": 0, "passthrough": 0,
             "companion": 0, "delta_applied": 0}
    d_samples: list[float] = []

    def flush_shard() -> None:
        nonlocal shard_id, buf, buf_bytes, total_size
        if not buf:
            return
        fname = f"merged-{shard_id:05d}.safetensors"
        save_file(buf, f"{args.out}/{fname}", metadata={"format": "pt"})
        for k in buf:
            index[k] = fname
        total_size += sum(t.untyped_storage().nbytes() for t in buf.values())
        print(f"[merge] shard {fname}: {len(buf)} tensors, "
              f"{buf_bytes / 2**30:.2f} GiB", flush=True)
        shard_id += 1
        buf, buf_bytes = {}, 0

    for shard_file in sorted(shard_tensors):
        with safe_open(f"{args.artifact}/{shard_file}", framework="pt",
                       device="cpu") as f:
            for name in shard_tensors[shard_file]:
                if name.endswith(COMPANIONS):
                    stats["companion"] += 1
                    continue
                base = name[:-7] if name.endswith(".weight") else None
                # companion detection via the GLOBAL index (companions may live
                # in a different shard than their .weight)
                if base is not None and f"{base}.weight_scale" in wm:
                    q = f.get_tensor(name)
                    s = load(f"{base}.weight_scale").float().reshape(-1, 1)
                    o_key = f"{base}.weight_offset"
                    o = (load(o_key).float().reshape(-1, 1)
                         if o_key in wm else torch.zeros(1))
                    if q.dtype == torch.int8 and q.shape[0] * 2 == s.shape[0]:
                        deq = (unpack_int4(q).float() * s + o)
                        kind = "expert"
                    else:
                        deq = q.float() * s + o
                        kind = "other"
                    tensor = deq.to(torch.bfloat16)
                    stats[f"dequant_{kind}"] += 1
                    m = EXPERT_RE.match(name)
                    if m:
                        layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                        d = delta_for(layer, expert, proj)
                        tensor = (tensor.float() + d).to(torch.bfloat16)
                        stats["delta_applied"] += 1
                        if len(d_samples) < 40:
                            d_samples.append(d.abs().mean().item())
                else:
                    tensor = f.get_tensor(name)
                    stats["passthrough"] += 1
                buf[name] = tensor.contiguous()
                buf_bytes += tensor.untyped_storage().nbytes()
                if buf_bytes >= max_bytes:
                    flush_shard()
    flush_shard()

    with open(f"{args.out}/model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": index}, f, indent=2)

    print(f"[merge] DONE: {len(index)} tensors, total_size={total_size / 2**30:.1f} GiB, "
          f"stats={stats}", flush=True)
    if d_samples:
        d_samples.sort()
        print(f"[merge] delta |mean| samples: min={d_samples[0]:.3e} "
              f"median={d_samples[len(d_samples)//2]:.3e} max={d_samples[-1]:.3e} "
              f"(n={len(d_samples)})", flush=True)


if __name__ == "__main__":
    main()
