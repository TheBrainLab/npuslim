#!/usr/bin/env python
"""Diagnostic: can the frozen-scale grid express the QAD LoRA delta?

For sample (layer, expert, proj): loads base artifact scale (frozen grid step s)
and the step-2000 LoRA delta; reports |delta| vs s and the energy preserved by
snapping delta onto the frozen grid: delta -> s*round(delta/s).

Decision value: frac(|delta| >= s/2) is the fraction of elements whose int4 code
flips; energy-kept near 1.0 means the frozen-scale requant preserves the QAD
correction well, near 0.0 means the correction is sub-grid and cannot survive.
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

import torch
from safetensors import safe_open

ART = f"{SHARED}/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234"
CKPT = f"{SHARED}/ylt/glm52-qad/ckpt_phase2_full/lora_step2000_p.pt"
SCALING = 4.0


def unpack_int4(p):
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
    wm = json.load(open(f"{ART}/model.safetensors.index.json"))["weight_map"]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    def load(name: str) -> torch.Tensor:
        with safe_open(f"{ART}/{wm[name]}", framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    cases = [(5, 0), (10, 50), (40, 100), (70, 0), (74, 255)]
    print(f"{'case':<24}{'RMS(delta)':>12}{'s_mean':>11}{'frac|d|>=s/2':>14}"
          f"{'E_kept':>9}{'d/s rms':>10}{'E_kept_g128':>13}{'E_kept_w8':>11}")
    for layer, exp in cases:
        p = f"model.layers.{layer}.mlp.experts.lora.params"
        bgu = sd[f"{p}.Bgu"][exp].float()
        agu = sd[f"{p}.Agu"][exp].float()
        dgu = (bgu @ agu) * SCALING
        bdn = sd[f"{p}.Bdn"][exp].float()
        adn = sd[f"{p}.Adn"][exp].float()
        ddn = (bdn @ adn) * SCALING
        half = dgu.shape[0] // 2
        for proj, delta in (("gate_proj", dgu[:half]), ("up_proj", dgu[half:]),
                            ("down_proj", ddn)):
            base = f"model.layers.{layer}.mlp.experts.{exp}.{proj}"
            s = load(f"{base}.weight_scale").float().reshape(-1, 1)
            q = unpack_int4(load(f"{base}.weight")).float()
            w_base = q * s                      # base dequant (on grid)
            w_merged = w_base + delta
            ratio = delta / s
            frac = (ratio.abs() >= 0.5).float().mean().item()
            resid = delta - s * torch.round(ratio)
            kept = 1 - (resid.norm() / delta.norm().clamp_min(1e-12)).item()
            rms = delta.pow(2).mean().sqrt().item()

            def grid_kept(w_m, delta_, group):
                """quantize w_m on a per-(row, col-group) symmetric grid
                (s_g = amax/7), return fraction of delta energy preserved as
                (snap(w_m) - w_base) vs delta."""
                out_cols = w_m.shape[1]
                ng = out_cols // group
                snap = torch.empty_like(w_m)
                for g in range(ng):
                    sl = slice(g * group, (g + 1) * group)
                    blk = w_m[:, sl]
                    s_g = blk.abs().amax(dim=1, keepdim=True) / 7.0
                    s_g = s_g.clamp_min(1e-12)
                    snap[:, sl] = s_g * torch.round(blk / s_g)
                corr = snap - w_base.expand_as(snap) if False else snap - w_base
                kept_g = 1 - ((corr - delta_).norm()
                              / delta_.norm().clamp_min(1e-12)).item()
                return kept_g

            kept_g = grid_kept(w_merged, delta, 128)
            s8 = w_merged.abs().amax(dim=1, keepdim=True) / 127.0
            snap8 = s8 * torch.round(w_merged / s8)
            corr8 = snap8 - w_base
            kept8 = 1 - ((corr8 - delta).norm()
                         / delta.norm().clamp_min(1e-12)).item()
            print(f"L{layer} E{exp} {proj:<11}{rms:>12.3e}{s.mean():>11.3e}"
                  f"{frac:>14.4f}{kept:>9.4f}{ratio.pow(2).mean().sqrt().item():>10.3f}"
                  f"{kept_g:>13.4f}{kept8:>11.4f}")


if __name__ == "__main__":
    main()
