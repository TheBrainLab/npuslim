"""GPTQGlmRotationAlgorithm 管线接线单测（E3-4）。

覆盖：配置/注册、_build_rotation、embed 折叠（chunk + 运行时模型双写）、
model.norm→lm_head 张量级折叠、FLOAT 快照同步、_collect_statistics 折叠顺序
（monkeypatch super 统计收集为 no-op）。
"""

import pytest
import torch
import torch.nn as nn

from npuslim.core import AlgorithmRegistry
from npuslim.qad.glm_rotation_quant import GPTQGlmRotationAlgorithm
from npuslim.qad.glm_rotation import get_glm52_rotate_maps
from npuslim.tasks.compressor.context import ChunkContext, LayerInfo, ModuleInfo

from tests.qad.test_glm_rotation import _FakeGlmLayer, _RMSNorm, _mini_rm


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 2
    first_k_dense_replace = 1
    q_lora_rank = 8
    kv_lora_rank = 4
    v_head_dim = 8
    qk_nope_head_dim = 4
    qk_rope_head_dim = 4


class _MiniInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([nn.Module(), _FakeGlmLayer(dense=False)])
        self.norm = _RMSNorm(16)


class _MiniModel(nn.Module):
    """顶层结构对齐 GlmMoeDsaForCausalLM：.model.{embed_tokens,layers,norm} + .lm_head。"""

    def __init__(self):
        super().__init__()
        self.config = _Cfg
        self.model = _MiniInner()
        self.lm_head = nn.Linear(16, 32, bias=False)


def _make_algo(**extra):
    kwargs = dict(wbits=4, group_size=-1, **extra)
    algo = GPTQGlmRotationAlgorithm(glm_rotation={"block_size": -1, "seed": 2026}, **kwargs)
    algo._runtime_model = _MiniModel()
    algo._build_rotation()
    return algo


def test_registration_and_config():
    cls = AlgorithmRegistry.get("GPTQGlmRotation")
    assert cls is GPTQGlmRotationAlgorithm
    algo = GPTQGlmRotationAlgorithm(wbits=4, group_size=-1)
    assert algo.glm_rotation_enabled is False
    algo2 = GPTQGlmRotationAlgorithm(wbits=4, group_size=-1, glm_rotation={"seed": 1})
    assert algo2.glm_rotation_enabled is True


def test_build_rotation_dims():
    algo = _make_algo()
    assert algo._rot_matrices["rot"].shape == (16, 16)
    assert algo._rot_matrices["rot_b_proj"].shape == (8, 8)
    assert algo._rot_params["num_layers"] == 2
    assert algo._rot_params["first_k_dense_replace"] == 1


def test_fold_pre_modules_embed():
    algo = _make_algo()
    model = algo._runtime_model.model
    w0 = model.embed_tokens.weight.data.clone()
    chunk = ChunkContext(
        chunk_index=0,
        pre_modules=[ModuleInfo(name="model.embed_tokens", tensors={"weight": w0.clone()})],
    )
    algo._fold_pre_modules(chunk)
    R = algo._rot_matrices["rot"]
    expected = (w0.float() @ R).to(w0.dtype)
    # chunk 快照
    assert torch.allclose(chunk.get_tensor("model.embed_tokens.weight"), expected, atol=1e-5)
    # 运行时模型
    assert torch.allclose(algo._runtime_model.model.embed_tokens.weight.data, expected, atol=1e-5)
    # 幂等标记（重复折叠由 _capture_initial_inputs 的 _embed_folded gating 防住）
    assert algo._embed_folded is True


def test_fold_post_modules_norm_lmhead():
    algo = _make_algo()
    gamma = torch.rand(16) * 2 + 0.5
    lm = torch.randn(32, 16)
    chunk = ChunkContext(
        chunk_index=1,
        post_modules=[
            ModuleInfo(name="model.norm", tensors={"weight": gamma.clone()}),
            ModuleInfo(name="lm_head", tensors={"weight": lm.clone()}),
        ],
    )
    algo._fold_post_modules(chunk)
    R = algo._rot_matrices["rot"]
    expected = ((lm.float() * gamma.view(1, -1)) @ R).to(lm.dtype)
    assert torch.allclose(chunk.get_tensor("model.norm.weight"), torch.ones(16), atol=1e-6)
    assert torch.allclose(chunk.get_tensor("lm_head.weight"), expected, atol=1e-4)
    assert algo._final_folded is True


def test_sync_layer_float_snapshots():
    """sync 必须把「折叠后」的运行时模块值写回 chunk（chunk 初始为折叠前值）。

    回归：旧实现遍历含 .weight 后缀会走到 Parameter 本身 → hasattr(weight) False
    → 静默跳过，chunk 保留未折叠原值（产物 router/kv_b_proj 未旋转）。
    """
    algo = _make_algo()
    layer = algo._runtime_model.model.layers[1]
    from npuslim.qad.glm_rotation import fold_glm52_layer

    layer_name = "model.layers.1"
    rels = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "mlp.gate.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.weights_proj.weight",
    ]

    def _mod(rel):
        m = layer
        for part in rel.rsplit(".", 1)[0].split("."):
            m = getattr(m, part)
        return m

    # chunk 初始 = 折叠前快照（模拟 loader 加载的原始张量）
    before = {rel: _mod(rel).weight.data.clone() for rel in rels}
    chunk = ChunkContext(
        chunk_index=0,
        layers=[LayerInfo(name=layer_name, index=1, tensors=before)],
    )
    # before 与 chunk.tensors 是同一 dict，update_tensor 会改写它 → 另存副本作断言基准
    before_ref = {rel: t.clone() for rel, t in before.items()}
    router_pre = before_ref["mlp.gate.weight"].clone()

    # 折叠运行时层（γ→1 + 旋转）
    fold_glm52_layer(layer, 1, algo._rot_matrices, **algo._rot_params)

    algo._current_chunk = chunk
    try:
        algo._sync_layer_float_snapshots(layer_name)
    finally:
        algo._current_chunk = None

    # norms 全部置 1（γ 已折进消费者）
    for n in ["input_layernorm", "post_attention_layernorm"]:
        t = chunk.get_tensor(f"{layer_name}.{n}.weight")
        assert torch.allclose(t, torch.ones_like(t), atol=1e-6), n
    # router / kv_b / indexer 与运行时模型折叠值一致，且 ≠ 折叠前原值（sync 真的写了）
    for rel in [
        "mlp.gate.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.weights_proj.weight",
    ]:
        after = chunk.get_tensor(f"{layer_name}.{rel}")
        assert torch.allclose(after, _mod(rel).weight.data, atol=0), rel
        assert not torch.equal(after, before_ref[rel]), f"sync 未生效: {rel}"
    assert not torch.equal(
        chunk.get_tensor(f"{layer_name}.mlp.gate.weight"), router_pre
    )


def test_collect_statistics_fold_order(monkeypatch):
    """_collect_statistics：先折叠 + 同步快照，再调父类统计收集（旋转空间）。"""
    algo = _make_algo()
    layer = algo._runtime_model.model.layers[1]

    called = {}

    def fake_super_collect(self, layer_module, handlers, *, layer_name, chunk_index):
        called["layer_name"] = layer_name
        # 统计收集时层权重应已折叠（norm γ=1）
        called["gamma_is_one"] = torch.allclose(
            layer.input_layernorm.weight.data, torch.ones_like(layer.input_layernorm.weight.data)
        )

    import npuslim.algorithms.quantization.gptq.gptq_algo as gmod

    monkeypatch.setattr(
        gmod.GPTQAlgorithm, "_collect_statistics", fake_super_collect
    )
    layer_name = "model.layers.1"
    chunk = ChunkContext(chunk_index=0, layers=[LayerInfo(name=layer_name, index=1, tensors={})])
    algo._current_chunk = chunk
    try:
        algo._collect_statistics(layer, {}, layer_name=layer_name, chunk_index=0)
    finally:
        algo._current_chunk = None
    assert called["layer_name"] == layer_name
    assert called["gamma_is_one"] is True
