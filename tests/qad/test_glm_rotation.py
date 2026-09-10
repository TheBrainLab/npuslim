"""GLM-5.2 残差流旋转折叠单测（E3）—— 原语 + 层折叠 fp 等价闸门。

关键验证：折叠后的网络函数与折叠前恒等（1e-3 量级），即
「norm 融合 + 旋转折叠」不改变模型输出。
"""

import math

import pytest
import torch
import torch.nn as nn

from npuslim.qad.glm_rotation import (
    fold_glm52_layer,
    fuse_ln_linear,
    get_glm52_rotate_maps,
    rotate_linear,
    rotate_weight_3d,
)
from npuslim.qad.rotation import QuaRotMode, create_rot


def _ortho_err(R):
    return (R @ R.t() - torch.eye(R.shape[0])).abs().max().item()


def test_rotate_linear_right_preserves_function():
    """right_rotate: W@R 与输入 x@R.T 相乘恢复原输出。"""
    torch.manual_seed(0)
    lin = nn.Linear(8, 4, bias=False)
    w0 = lin.weight.data.clone()
    R = create_rot(QuaRotMode.HADAMARD, 8, block_size=-1, seed=7)
    x = torch.randn(3, 8)
    y0 = lin(x)
    rotate_linear(lin, R, right_rotate=True)  # W ← W @ R
    y1 = lin(x @ R)                            # 输入按行向量右乘 R = 折叠补偿
    assert torch.allclose(y0, y1, atol=1e-3)


def test_rotate_linear_left_preserves_function():
    """left_rotate: R.T@W 使输出旋转 R@y。"""
    torch.manual_seed(1)
    lin = nn.Linear(8, 4, bias=True)
    b0 = lin.bias.data.clone()
    R = create_rot(QuaRotMode.HADAMARD, 4, block_size=-1, seed=7)
    x = torch.randn(3, 8)
    y0 = lin(x)
    rotate_linear(lin, R, right_rotate=False)  # W ← R.T@W, b ← R.T@b
    y1 = lin(x)
    assert torch.allclose(y0 @ R, y1, atol=1e-3)  # 输出行向量右乘 R
    assert not torch.equal(lin.bias.data, b0)


def test_rotate_weight_3d_right():
    """3D 专家参数 right 折叠（输入侧 in 维）逐专家一致。"""
    torch.manual_seed(2)
    R = create_rot(QuaRotMode.HADAMARD, 8, block_size=-1, seed=7)
    w3d = nn.Parameter(torch.randn(2, 4, 8))
    w0 = w3d.data.clone()
    rotate_weight_3d(w3d, R, right_rotate=True)
    for e in range(2):
        expected = w0[e].float() @ R
        assert torch.allclose(w3d.data[e].float(), expected, atol=1e-3)


def test_fuse_ln_linear_preserves_function():
    """γ 折进线性后，norm 置 1 仍保持函数。"""
    torch.manual_seed(3)
    class _LN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(d) * 2 + 0.5)
        def forward(self, x):
            return x * self.weight
    ln = _LN(8)
    lin = nn.Linear(8, 4, bias=False)
    w0 = lin.weight.data.clone()
    gamma0 = ln.weight.data.clone()
    x = torch.randn(3, 8)
    y0 = lin(ln(x))
    fuse_ln_linear([ln], [lin])
    assert torch.equal(ln.weight.data, torch.ones(8))  # γ → 1
    assert not torch.equal(lin.weight.data, w0)
    y1 = lin(ln(x))
    assert torch.allclose(y0, y1, atol=1e-3)


class _RMSNorm(nn.Module):
    """RMSNorm（无均值减法，对特征旋转不变——与真实 GLM 一致）。"""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(d) * 2 + 0.5)

    def forward(self, x):
        return self.weight * x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


class _FakeGlmLayer(nn.Module):
    """维度一致的迷你 GLM-MLA 层，匹配旋转命令 dims。

    H=16(rot) / q_lora=8(rot_b) / kv_lora=4(rot_kv) / rope=4 / nope=4 / v=8(rot_uv)
    结构与真实 GLM-5.2 MLA 一致：q_a→lora 空间→q_b；kv_a→[c,rope]→kv_b→[nope,v]→o。
    注意：必须用 RMSNorm（LayerNorm 的均值减法破坏旋转不变性）。
    """

    def __init__(self, H=16, q_lora=8, kv_lora=4, rope=4, nope=4, v=8, num_exp=2, dense=False):
        super().__init__()
        self.H, self.q_lora, self.kv_lora, self.rope, self.nope, self.v = H, q_lora, kv_lora, rope, nope, v
        self.input_layernorm = _RMSNorm(H)
        self.post_attention_layernorm = _RMSNorm(H)
        self.self_attn = nn.Module()
        self.self_attn.q_a_proj = nn.Linear(H, q_lora, bias=False)
        self.self_attn.kv_a_proj_with_mqa = nn.Linear(H, kv_lora + rope, bias=False)
        self.self_attn.q_a_layernorm = _RMSNorm(q_lora)
        self.self_attn.kv_a_layernorm = _RMSNorm(kv_lora)
        self.self_attn.q_b_proj = nn.Linear(q_lora, H, bias=False)
        self.self_attn.kv_b_proj = nn.Linear(kv_lora, nope + v, bias=False)
        self.self_attn.o_proj = nn.Linear(v, H, bias=False)
        # DSA indexer（n_heads=2, head_dim=4）：wq_b 吃 lora 空间、wk/weights_proj 吃 H 空间
        self.self_attn.indexer = nn.Module()
        self.self_attn.indexer.wq_b = nn.Linear(q_lora, 8, bias=False)
        self.self_attn.indexer.wk = nn.Linear(H, 8, bias=False)
        self.self_attn.indexer.weights_proj = nn.Linear(H, 2, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(H, num_exp, bias=False)
        self.dense = dense
        if dense:
            self.mlp.gate_proj = nn.Linear(H, 8, bias=False)
            self.mlp.up_proj = nn.Linear(H, 8, bias=False)
            self.mlp.down_proj = nn.Linear(8, H, bias=False)
        else:
            I = 8
            self.mlp.experts = nn.Module()
            self.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(num_exp, 2 * I, H))
            self.mlp.experts.down_proj = nn.Parameter(torch.randn(num_exp, H, I))
            self.mlp.shared_experts = nn.Module()
            self.mlp.shared_experts.gate_proj = nn.Linear(H, I, bias=False)
            self.mlp.shared_experts.up_proj = nn.Linear(H, I, bias=False)
            self.mlp.shared_experts.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        H, kv_lora, rope, nope, v = self.H, self.kv_lora, self.rope, self.nope, self.v
        h = self.input_layernorm(x)
        qa = self.self_attn.q_a_proj(h)
        kva = self.self_attn.kv_a_proj_with_mqa(h)
        qb = self.self_attn.q_b_proj(self.self_attn.q_a_layernorm(qa))          # 内部（注意力），不进残差
        kv_b = self.self_attn.kv_b_proj(self.self_attn.kv_a_layernorm(kva[..., :kv_lora]))
        o = self.self_attn.o_proj(kv_b[..., nope:])                            # 注意力输出
        resid = x + o                                                          # 残差加原始输入 + attn_out
        h2 = self.post_attention_layernorm(resid)
        r = self.mlp.gate(h2)
        if self.dense:
            gu = torch.cat([self.mlp.gate_proj(h2), self.mlp.up_proj(h2)], dim=-1)
            gate, up = gu.chunk(2, dim=-1)
            mlp_out = self.mlp.down_proj(torch.nn.functional.silu(gate) * up)
        else:
            top = r.argmax(dim=-1)
            gu = self.mlp.experts.gate_up_proj[top]  # [B, 2I, H]
            mid = []
            for b in range(x.shape[0]):
                g, u = gu[b].chunk(2, dim=0)
                mid.append(torch.nn.functional.silu(h2[b] @ g.t()) * (h2[b] @ u.t()))
            mid = torch.stack(mid)
            sh = torch.nn.functional.silu(self.mlp.shared_experts.gate_proj(h2)) * self.mlp.shared_experts.up_proj(h2)
            mlp_out = self.mlp.shared_experts.down_proj(sh) + torch.stack(
                [self.mlp.experts.down_proj[top[b]] @ mid[b] for b in range(x.shape[0])])
        return resid + mlp_out, r


def _mini_rm(H=16, q_lora=8, kv_lora=4, rope=4, nope=4, v=8):
    return {
        "rot": create_rot(QuaRotMode.HADAMARD, H, block_size=-1, seed=7),
        "rot_b_proj": create_rot(QuaRotMode.HADAMARD, q_lora, block_size=-1, seed=7),
        "rot_uv": create_rot(QuaRotMode.HADAMARD, v, block_size=-1, seed=7),
        "rot_kv_b_proj": create_rot(QuaRotMode.HADAMARD, kv_lora, block_size=-1, seed=7),
    }


def test_fold_glm52_layer_fp_equivalence_routed():
    """路由层折叠后网络函数在旋转空间恒等（fp 等价闸门）。

    残差流方案：折叠后层接收旋转输入 x@R，输出亦被旋转 y@R。
    layer_folded(x@R) == layer_orig(x)@R
    """
    torch.manual_seed(4)
    layer = _FakeGlmLayer(dense=False)
    rm = _mini_rm()
    x = torch.randn(3, 16)
    with torch.no_grad():
        y0, r0 = layer(x)
        touched = fold_glm52_layer(layer, 1, rm, num_layers=2, first_k_dense_replace=1, num_experts=2, qk_nope_head_dim=4, qk_rope_head_dim=4)
        y1, r1 = layer(x @ rm["rot"])
    assert len(touched) > 0
    assert torch.allclose(y0 @ rm["rot"], y1, atol=1e-2, rtol=1e-2)


def test_fold_glm52_layer_fp_equivalence_dense():
    """dense 层折叠后网络函数在旋转空间恒等。"""
    torch.manual_seed(5)
    layer = _FakeGlmLayer(dense=True)
    rm = _mini_rm()
    x = torch.randn(2, 16)
    with torch.no_grad():
        y0, r0 = layer(x)
        fold_glm52_layer(layer, 1, rm, num_layers=2, first_k_dense_replace=2, num_experts=2, qk_nope_head_dim=4, qk_rope_head_dim=4)
        y1, r1 = layer(x @ rm["rot"])
    assert torch.allclose(y0 @ rm["rot"], y1, atol=1e-2, rtol=1e-2)


def test_glm52_rotate_maps_orthogonal():
    pre_run, cmds, rm = get_glm52_rotate_maps(num_layers=2)
    for name, R in rm.items():
        assert _ortho_err(R) < 1e-3, name
    # rot 用于 6144 全维（Kronecker 路径）
    assert rm["rot"].shape == (6144, 6144)
    assert rm["rot_b_proj"].shape == (2048, 2048)
    assert rm["rot_uv"].shape == (256, 256)
    assert rm["rot_kv_b_proj"].shape == (512, 512)
    assert len(pre_run) == 1 and pre_run[0].target == "embed_tokens"


class _MiniGlm(nn.Module):
    """embed + 一层 + model.norm + lm_head 的迷你 GLM（验证 pre_run/post_run 折叠）。"""

    def __init__(self, H=16, vocab=32):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, H)
        self.model.norm = _RMSNorm(H)
        self.lm_head = nn.Linear(H, vocab, bias=False)
        self.layer = _FakeGlmLayer(H=H, dense=True)
        # 用真实层路径挂载：model.layers.1 = layer
        self.model.layers = nn.ModuleList([nn.Module(), self.layer])

    def forward(self, ids):
        x = self.model.embed_tokens(ids)
        x, _ = self.layer(x)
        return self.lm_head(self.model.norm(x))


def test_fold_embed_and_final_equivalence():
    """embed/lm_head 折叠后整网函数恒等：embed@R → ... → lm_head 吸收 R。"""
    from npuslim.qad.glm_rotation import fold_embed_and_final

    import copy
    torch.manual_seed(6)
    rm = _mini_rm()
    m0 = _MiniGlm()
    m1 = copy.deepcopy(m0)
    ids = torch.randint(0, 32, (3, 5))
    with torch.no_grad():
        y0 = m0(ids)
        # 折叠 embed + layer + lm_head（layer 需先折叠，与 fold_glm52_layer 一致）
        fold_embed_and_final(m1, rm, layer_name_prefix="model")
        fold_glm52_layer(m1.layer, 1, rm, num_layers=2, first_k_dense_replace=2,
                         num_experts=2, qk_nope_head_dim=4, qk_rope_head_dim=4)
        # 输入 id → embed 输出已在旋转空间（fold_embed 后）；直接走 m1 即整网折叠
        y1 = m1(ids)
    assert torch.allclose(y0, y1, atol=1e-2, rtol=1e-2)
    # 验证 model.norm γ → 1（quarot ln 融合）
    assert torch.allclose(m1.model.norm.weight.data, torch.ones(16), atol=1e-6)


def test_fold_indexer_invariance():
    """DSA indexer 折叠不变性：wq_b/wk/weights_proj 折叠后输出值逐元素保持。

    adapter 旋转图含 indexer 三件套（model_adapter.py:407-415）：
    wk/weights_proj right rot（消费 R 空间 norm 输出）+ input_layernorm γ 融合；
    wq_b right rot_b（消费 rot_b lora 空间）+ q_a_layernorm γ 融合。
    缺任何一件 indexer 分数变化 → top-k 稀疏选择改变。
    """
    torch.manual_seed(7)
    rm = _mini_rm()
    m0 = _FakeGlmLayer(dense=False)
    import copy
    m1 = copy.deepcopy(m0)
    x = torch.randn(3, 16)
    R = rm["rot"]

    def indexer_signals(m, x):
        h = m.input_layernorm(x)
        qa_ln = m.self_attn.q_a_layernorm(m.self_attn.q_a_proj(h))
        q = m.self_attn.indexer.wq_b(qa_ln)
        k = m.self_attn.indexer.wk(h)
        w = m.self_attn.indexer.weights_proj(h)
        return q, k, w

    with torch.no_grad():
        s0 = indexer_signals(m0, x)
        fold_glm52_layer(m1, 1, rm, num_layers=2, first_k_dense_replace=1,
                         num_experts=2, qk_nope_head_dim=4, qk_rope_head_dim=4)
        s1 = indexer_signals(m1, x @ R)
    for name, a, b in zip(["wq_b", "wk", "weights_proj"], s0, s1):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), f"{name} diff {(a-b).abs().max().item()}"


def test_rotate_weight_3d_chunked_equivalent():
    """分块折叠（E=20 > _EXPERT_FOLD_CHUNK=16，循环边界 16+4）== 整块 fp32 计算。

    回归：全量 GLM gate_up_proj [256,4096,6144] 整块 .float() = 24GiB → NPU OOM。
    """
    torch.manual_seed(6)
    E, out, ins = 20, 8, 16
    w0 = torch.randn(E, out, ins)
    R = create_rot(QuaRotMode.HADAMARD, ins, block_size=-1, seed=7)
    Rl = create_rot(QuaRotMode.HADAMARD, out, block_size=-1, seed=8)
    wr = nn.Parameter(w0.clone())
    rotate_weight_3d(wr, R, right_rotate=True)
    assert torch.allclose(wr.data.float(), w0.float() @ R, atol=1e-4)
    wl = nn.Parameter(w0.clone())
    rotate_weight_3d(wl, Rl, right_rotate=False)
    assert torch.allclose(wl.data.float(), Rl.T @ w0.float(), atol=1e-4)


def test_fold_glm52_layer_chunked_experts_values():
    """E=20（> _EXPERT_FOLD_CHUNK=16，循环边界 16+4）：直接校验分块 3D 折叠值。

    不走 forward——假层用 r.argmax 选专家，E 大时 top-2 分差极小、argmax 易翻转，
    会误报为折叠错误。这里对每个专家断言 gate_up/down 的 gamma 融合+旋转精确值。
    """
    import copy

    torch.manual_seed(7)
    layer = _FakeGlmLayer(num_exp=20, dense=False)
    rm = _mini_rm()
    R = rm["rot"]
    gu0 = layer.mlp.experts.gate_up_proj.data.clone()
    dn0 = layer.mlp.experts.down_proj.data.clone()
    gamma_post = layer.post_attention_layernorm.weight.data.clone()
    layer2 = copy.deepcopy(layer)
    fold_glm52_layer(
        layer2, 1, rm, num_layers=2, first_k_dense_replace=1, num_experts=20,
        qk_nope_head_dim=4, qk_rope_head_dim=4,
    )
    # gate_up_proj: post_attn γ 融合（输入列 H 维）+ right rot
    for e in range(20):
        expected_gu = (gu0[e].float() * gamma_post.view(1, -1)) @ R
        assert torch.allclose(layer2.mlp.experts.gate_up_proj.data[e].float(), expected_gu, atol=1e-4), f"gate_up[{e}]"
    # down_proj: left rot（输出侧）
    for e in range(20):
        expected_dn = R.T @ dn0[e].float()
        assert torch.allclose(layer2.mlp.experts.down_proj.data[e].float(), expected_dn, atol=1e-4), f"down[{e}]"
    # post_attention_layernorm γ → 1
    assert torch.allclose(layer2.post_attention_layernorm.weight.data, torch.ones_like(gamma_post), atol=1e-6)
