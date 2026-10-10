"""QAD Phase-1：从 P1 部署产物读取专家 SSZ per-output-channel scale，供 STE fake-quant 对齐。

背景（0908 §4.8-F 注 / §5.13）：GLM QAD 的学生 fake-quant 必须精确复刻部署期反量化，
其中 expert 的 weight_scale 取部署产物（SSZ = RTN + 迭代 LSQ 精修）的 scale，而非训练时
现场 minmax RTN。P1 = L0（SSZ+RTN，无 AWQ/SQ/旋转），故专家 scale 即最终部署 scale。

产物布局（per-expert 2D，见 docs §1.5）：
  model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale  [I,1]
  model.layers.{L}.mlp.experts.{e}.up_proj.weight_scale    [I,1]
  model.layers.{L}.mlp.experts.{e}.down_proj.weight_scale  [H,1]

训练侧专家为 fused 3D：
  experts.gate_up_proj [E, 2I, H]（HF 约定 [gate; up] 沿输出维拼接）
  experts.down_proj    [E, H, I]
→ 返回 {"gate_up": [E, 2I, 1], "down": [E, H, 1]}，直接喂给
  fake_quant_weight(w, bits=4, scale=s)。
"""
from __future__ import annotations

import json
import os

import torch


def _load_index(artifact_dir: str) -> dict:
    for fn in ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json"):
        p = os.path.join(artifact_dir, fn)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["weight_map"]
    raise FileNotFoundError(f"no safetensors index in {artifact_dir}")


def load_expert_ssz_scales(artifact_dir: str, layer_idx: int, num_experts: int,
                           cache: dict | None = None) -> dict:
    """返回 {"gate_up": [E, 2I, 1], "down": [E, H, 1]}（float32），用于 STE fake-quant。"""
    if cache is not None and layer_idx in cache:
        return cache[layer_idx]
    from safetensors import safe_open

    idx = _load_index(artifact_dir)
    by_shard: dict = {}
    for e in range(num_experts):
        base = f"model.layers.{layer_idx}.mlp.experts.{e}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            n = f"{base}.{proj}.weight_scale"
            if n not in idx:
                raise KeyError(f"missing {n} in {artifact_dir}")
            by_shard.setdefault(idx[n], []).append((e, proj, n))
    vals: dict = {}
    for shard, items in by_shard.items():
        with safe_open(os.path.join(artifact_dir, shard), framework="pt") as f:
            for e, proj, n in items:
                vals[(e, proj)] = f.get_tensor(n).float().reshape(-1)
    gate = torch.stack([vals[(e, "gate_proj")] for e in range(num_experts)], 0)
    up = torch.stack([vals[(e, "up_proj")] for e in range(num_experts)], 0)
    dn = torch.stack([vals[(e, "down_proj")] for e in range(num_experts)], 0)
    out = {"gate_up": torch.cat([gate, up], dim=1).unsqueeze(-1),   # [E,2I,1] = [gate; up]
           "down": dn.unsqueeze(-1)}                                # [E,H,1]
    if cache is not None:
        cache[layer_idx] = out
    return out


def make_scale_provider(artifact_dir: str, num_experts: int):
    """返回 layer_idx -> {"gate_up","down"} 的带缓存 provider（供 wrap_all_experts_glm 用）。"""
    cache: dict = {}

    def provider(layer_idx: int) -> dict:
        return load_expert_ssz_scales(artifact_dir, layer_idx, num_experts, cache=cache)

    return provider
