"""QAD Phase-1 loader 单测：deploy_scales.load_expert_ssz_scales 的形状/融合/取值/缓存。

运行（e6 容器，CPU）：
    PYTHONPATH=/llm_infer_workspace/ylt/npuslim/src python3 tests/qad/test_deploy_scales.py
用合成 safetensors 产物，无需真实模型/卡片。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.deploy_scales import load_expert_ssz_scales, make_scale_provider  # noqa: E402

PROJS = (("gate_proj", "I"), ("up_proj", "I"), ("down_proj", "H"))


def _make_artifact(root: str, layer: int, E: int, I: int, H: int) -> dict:
    tensors, weight_map = {}, {}
    for e in range(E):
        for proj, dim in PROJS:
            n = I if dim == "I" else H
            name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale"
            tensors[name] = torch.rand(n, 1)
            weight_map[name] = "shard.safetensors"
    save_file(tensors, os.path.join(root, "shard.safetensors"))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return tensors


def test_loader_fusion_and_cache() -> None:
    L, E, I, H = 2, 4, 8, 16
    with tempfile.TemporaryDirectory() as d:
        t = _make_artifact(d, L, E, I, H)
        s = load_expert_ssz_scales(d, L, E)
        assert tuple(s["gate_up"].shape) == (E, 2 * I, 1), tuple(s["gate_up"].shape)
        assert tuple(s["down"].shape) == (E, H, 1), tuple(s["down"].shape)
        for e in range(E):
            g = t[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale"].reshape(-1)
            u = t[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight_scale"].reshape(-1)
            dn = t[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight_scale"].reshape(-1)
            assert torch.allclose(s["gate_up"][e, :I, 0], g)
            assert torch.allclose(s["gate_up"][e, I:2 * I, 0], u)
            assert torch.allclose(s["down"][e, :, 0], dn)
        prov = make_scale_provider(d, E)
        assert prov(L) is prov(L)
    print("deploy_scales loader ALL PASS (fusion + cache)")


if __name__ == "__main__":
    test_loader_fusion_and_cache()
