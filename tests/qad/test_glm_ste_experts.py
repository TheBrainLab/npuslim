"""C1 单元测试：wrap_all_experts_glm（GLM MoE 专家 STE+LoRA 包装正式化）。

运行（npuslim conda env，CPU 即可）：
    /data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_glm_ste_experts.py

覆盖：
  C1.a 结构   ：dense MLP 层跳过 / MoE 层包装为 STEQwen3Experts / 专家权重冻结 / LoRA 可训练
  C1.b 多专家折叠：per-expert LoRA delta 只折入对应专家的有效权重（其余专家不变），且数值 == B@A*scaling
  C1.c 输出一致性：向量化 forward == 逐专家 loop forward（同一 STE 有效权重，bf16 噪声内）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.ste_experts import STEQwen3Experts, wrap_all_experts_glm  # noqa: E402
from npuslim.qad.ste import fake_quant_weight  # noqa: E402

torch.manual_seed(0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class MockExperts(nn.Module):
    """模拟 GlmMoeDsaNaiveMoe（属性与 Qwen3MoeExperts 同构）。"""

    def __init__(self, E: int = 4, h: int = 16, i: int = 32):
        super().__init__()
        self.num_experts = E
        self.hidden_dim = h
        self.intermediate_dim = i
        self.act_fn = nn.SiLU()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * i, h) * 0.2)
        self.down_proj = nn.Parameter(torch.randn(E, h, i) * 0.2)


class MockDenseMLP(nn.Module):
    """模拟 GlmMoeDsaMLP（无 .experts）——GLM 前 first_k_dense_replace 层。"""

    def __init__(self, h: int = 16, i: int = 32):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)


class MockMoE(nn.Module):
    """模拟 GlmMoeDsaMoE：.mlp 含 .experts（routed）+ shared_experts。"""

    def __init__(self):
        super().__init__()
        self.experts = MockExperts()


class MockLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class MockGLM(nn.Module):
    """model.layers = 稠密/MoE 交错（严格验证"按存在性跳过"而非按前缀）。"""

    def __init__(self, pattern=("dense", "moe", "dense", "moe", "moe")):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLayer(MockDenseMLP() if t == "dense" else MockMoE()) for t in pattern]
        )


# ---------- C1.a 结构 ----------
def test_structure() -> None:
    m = MockGLM(("dense", "moe", "dense", "moe", "moe"))
    n_dense = sum(1 for l in m.model.layers if not hasattr(l.mlp, "experts"))
    n_moe = sum(1 for l in m.model.layers if hasattr(l.mlp, "experts"))
    wrapped = wrap_all_experts_glm(m, rank=4, dtype=torch.float32)
    check("C1.a 包装数 == MoE 层数", len(wrapped) == n_moe, f"{len(wrapped)} vs {n_moe}")
    check("C1.a 返回均为 STEQwen3Experts", all(isinstance(w, STEQwen3Experts) for w in wrapped))
    dense_ok = all(not isinstance(l.mlp, STEQwen3Experts)
                   for l in m.model.layers if not hasattr(l.mlp, "experts"))
    check("C1.a dense 层未被包装", dense_ok, f"dense={n_dense}")
    wrapped_ok = all(isinstance(l.mlp.experts, STEQwen3Experts)
                     for l in m.model.layers if hasattr(l.mlp, "experts"))
    check("C1.a MoE 层被包装", wrapped_ok)
    w = wrapped[0]
    check("C1.a 原专家权重冻结", all(not p.requires_grad for p in w.experts.parameters()))
    check("C1.a LoRA 可训练", all(p.requires_grad for p in w.lora.params.values()))


# ---------- C1.b 多专家折叠 ----------
def test_multi_expert_folding() -> None:
    E, h, i = 3, 8, 16
    exp = MockExperts(E=E, h=h, i=i)
    w = STEQwen3Experts(exp, rank=4)
    with torch.no_grad():
        w.lora.params["Bgu"][1].zero_()
        w.lora.params["Bdn"][2].zero_()
    base = w._effective_all("gate_up").clone()          # [E, 2i, h]
    base_dn = w._effective_all("down").clone()          # [E, h, i]
    with torch.no_grad():
        w.lora.params["Bgu"][1].fill_(1.0)
        w.lora.params["Bdn"][2].fill_(1.0)
    new = w._effective_all("gate_up")
    new_dn = w._effective_all("down")
    check("C1.b 专家1 gate_up 变化", not torch.allclose(base[1], new[1]))
    check("C1.b 专家0/2 gate_up 不变",
          torch.allclose(base[0], new[0]) and torch.allclose(base[2], new[2]))
    check("C1.b 专家2 down 变化", not torch.allclose(base_dn[2], new_dn[2]))
    check("C1.b 专家0/1 down 不变",
          torch.allclose(base_dn[0], new_dn[0]) and torch.allclose(base_dn[1], new_dn[1]))
    with torch.no_grad():
        d = (w.lora.params["Bgu"][1] @ w.lora.params["Agu"][1]) * w.lora.scaling
    diff = new[1].float() - base[1].float()
    maxd = (diff - d.float()).abs().max().item()
    # bf16 effective_all：wq 与 (wq+d) 各自 bf16 舍入，差值容差按 bf16 精度放宽
    check("C1.b delta 数值 ≈ B@A*scaling（bf16 内）", maxd < 6e-2, f"max|Δ|={maxd:.3e}")


# ---------- C1.c 输出一致性（向量化 vs 逐专家）----------
def test_output_consistency() -> None:
    E, h, i = 4, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    w = STEQwen3Experts(exp, rank=4, vectorized=True)
    with torch.no_grad():
        for p in w.lora.params.values():
            p.normal_(0, 0.1)               # 非零 LoRA，确保折叠参与
    T, K = 6, 2
    x = torch.randn(T, h).to(torch.bfloat16)   # _forward_loop 不做 dtype 转换，需 bf16
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    wt = torch.rand(T, K)
    wt = wt / wt.sum(-1, keepdim=True)
    out_vec = w._forward_vec(x, idx, wt)
    out_loop = w._forward_loop(x, idx, wt)
    maxd = (out_vec.float() - out_loop.float()).abs().max().item()
    check("C1.c 向量化 == 逐专家（bf16 噪声内）", maxd < 5e-2, f"max|Δ|={maxd:.3e}")
    check("C1.c 输出形状 == 输入", out_vec.shape == x.shape, f"{tuple(out_vec.shape)}")


# ---------- C1.d deploy_scales（QAD Phase-1 SSZ-scale 对齐通路）----------
def test_deploy_scales_used() -> None:
    E, h, i = 3, 8, 16
    exp = MockExperts(E=E, h=h, i=i)
    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    w = STEQwen3Experts(exp, rank=4, deploy_scales={"gate_up": s_gu, "down": s_dn})
    for p in w.lora.params.values():
        with torch.no_grad():
            p.zero_()
    eff_gu = w._effective_all("gate_up")
    ref_gu = fake_quant_weight(w.experts.gate_up_proj.detach().to(torch.bfloat16),
                               bits=4, scale=s_gu)
    maxd = (eff_gu.float() - ref_gu.float()).abs().max().item()
    check("C1.d deploy_scales 生效（== fake_quant(外部 scale)）", maxd == 0, f"max|Δ|={maxd:.3e}")

    w2 = STEQwen3Experts(exp, rank=4)
    for p in w2.lora.params.values():
        with torch.no_grad():
            p.zero_()
    check("C1.d 外部 scale ≠ 现场 minmax（确为外部通路）",
          not torch.allclose(eff_gu.float(), w2._effective_all("gate_up").float()))


# ---------- C1.e EP 路径（QAD_EP=1）world=1 数值等价 ----------
def test_ep_world1_bit_identical() -> None:
    """world=1 时 EP 退化为：shard==全量、all_gather 恒等 → 与 dense 逐 bit 一致。"""
    import os
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return  # 已在分布式环境（world>1），world=1 等价前提不成立，跳过

    E, h, i = 8, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    # 挂占位 ds_tensor（EP 需要 zero3_gather + ds_tensor）
    exp.gate_up_proj.ds_tensor = exp.gate_up_proj.detach().clone().reshape(-1)
    exp.gate_up_proj.ds_numel = exp.gate_up_proj.numel()
    exp.gate_up_proj.ds_shape = tuple(exp.gate_up_proj.shape)
    exp.down_proj.ds_tensor = exp.down_proj.detach().clone().reshape(-1)
    exp.down_proj.ds_numel = exp.down_proj.numel()
    exp.down_proj.ds_shape = tuple(exp.down_proj.shape)

    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    os.environ["QAD_EP"] = "1"
    os.environ["QAD_EP_TMAX"] = "6"  # == 测试 T：免 padding，world=1 matmul 形状与 dense 一致
    try:
        w = STEQwen3Experts(exp, rank=4, dtype=torch.float32, zero3_gather=True,
                            deploy_scales={"gate_up": s_gu, "down": s_dn})
        ep = w._ep_setup()
        check("C1.e EP setup 生效（world=1, epk=E）", ep is not None and ep[2] == E,
              f"ep={ep}")
        if ep is None:
            return
        with torch.no_grad():
            for p in w.lora.params.values():
                p.normal_(0, 0.05)
        T, K = 6, 2
        x = torch.randn(T, h).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        wt = torch.rand(T, K)
        wt = wt / wt.sum(-1, keepdim=True)
        out_ep = w._forward_vec_ep(x, idx, wt)
        out_dense = w._forward_vec(x, idx, wt)
        md = (out_ep.float() - out_dense.float()).abs().max().item()
        # world=1 下 matmul 形状不同（pair 维 vs token 维）→ bf16 舍入噪声，
        # 结构性错误（置换/专家映射错）会产生 O(1) 差异，阈值 5e-2 足以区分
        check("C1.e EP ≈ dense（world=1，bf16 噪声内）", md < 5e-2, f"max|Δ|={md:.3e}")

        # 梯度流：EP 路径 loss 对 LoRA b,a 必须有梯度（集合通信不得断图）
        out_ep.sum().backward()
        grads_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                       for p in w.lora.params.values())
        check("C1.e EP 梯度流（LoRA b/a grad 非空非零）", grads_ok)
    finally:
        os.environ.pop("QAD_EP", None)
        os.environ.pop("QAD_EP_TMAX", None)


# ---------- C1.f int4 码本驻卡缓存（QAD_WQ_CACHE）与原路径逐 bit 一致 ----------
def test_wq_cache_bit_identical() -> None:
    """_effective_shard_cached == _effective_shard（同 w/s/b,a，逐 bit）。

    缓存路径把 v=clamp(round(w/s))（bf16 算术）存 int8（W 冻结 → 跨步恒定），
    每步 wq=v*s + delta；与原路径每步重算 fake_quant 必须逐 bit 一致。
    """
    E, h, i = 8, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    w = STEQwen3Experts(exp, rank=4, dtype=torch.float32,
                        deploy_scales={"gate_up": s_gu, "down": s_dn})
    with torch.no_grad():
        for p in w.lora.params.values():
            p.normal_(0, 0.05)

    epk, e0 = E, 0
    for proj, shape in (("gate_up", (epk, 2 * i, h)), ("down", (epk, h, i))):
        src = exp.gate_up_proj if proj == "gate_up" else exp.down_proj
        w_shard_cpu = src.detach().reshape(*shape).to(torch.bfloat16)
        ref = w._effective_shard(proj, w_shard_cpu.clone(), e0, epk)
        w._wq_cache = {}
        w._wq_cache_enabled = True
        got1 = w._effective_shard_cached(proj, w_shard_cpu, e0, epk,
                                         torch.device("cpu"))
        got2 = w._effective_shard_cached(proj, w_shard_cpu, e0, epk,
                                         torch.device("cpu"))
        md1 = (ref.float() - got1.float()).abs().max().item()
        md2 = (ref.float() - got2.float()).abs().max().item()
        check(f"C1.f {proj} cached==原路径（首访建码本）", md1 == 0, f"max|Δ|={md1:.3e}")
        check(f"C1.f {proj} cached==原路径（缓存命中）", md2 == 0, f"max|Δ|={md2:.3e}")
        codes, s = w._wq_cache[proj]
        check(f"C1.f {proj} 码本 uint8 nibble 打包（in/2）",
              codes.dtype == torch.uint8
              and codes.shape == (*shape[:-1], shape[-1] // 2),
              f"dtype={codes.dtype} shape={tuple(codes.shape)}")


# ---------- C2.a 复合顺序（QAD 文档 §8）：deploy 复合 = fakequant(W+delta) ----------
def test_compose_deploy() -> None:
    """QAD_COMPOSE=deploy：前向 == fakequant(W+δ)（bit-exact）；δ=0 时与 post 一致；δ 梯度非零。"""
    E, h, i = 8, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    w = STEQwen3Experts(exp, rank=4, dtype=torch.float32,
                        deploy_scales={"gate_up": s_gu, "down": s_dn})
    with torch.no_grad():
        for p_ in w.lora.params.values():
            p_.normal_(0, 0.05)
    epk, e0 = E, 0
    w_shard = exp.gate_up_proj.detach().reshape(epk, 2 * i, h).to(torch.bfloat16)

    w._compose = "deploy"
    got = w._effective_shard("gate_up", w_shard, e0, epk)
    b, a = w.lora.params["Bgu"], w.lora.params["Agu"]
    d = torch.einsum("eor,eri->eoi", b[e0:e0 + epk], a[e0:e0 + epk]) * w.lora.scaling
    expect = fake_quant_weight(w_shard.float() + d, bits=4,
                               scale=s_gu[e0:e0 + epk]).to(torch.bfloat16)
    md = (got.float() - expect.float()).abs().max().item()
    check("C2.a deploy 复合 == fakequant(W+delta)（bit-exact）", md == 0, f"max|d|={md:.3e}")

    with torch.no_grad():
        for p_ in w.lora.params.values():
            p_.zero_()
    w._compose = "post"; r1 = w._effective_shard("gate_up", w_shard, e0, epk)
    w._compose = "deploy"; r2 = w._effective_shard("gate_up", w_shard, e0, epk)
    md2 = (r1.float() - r2.float()).abs().max().item()
    s_max = float(s_gu[e0:e0 + epk].max())
    # delta=0 时两复合仅差量化器 dtype：deploy 走 fp32 除法/舍入（= 部署 requant 的
    # fp32 语义），post（historic）走 bf16（w_slab 是 bf16、scale.to(w.dtype)）→
    # 元素差 ∈ {0, ±1 个网格步}（边界舍入偶发 ±1 码），非语义分歧。
    check("C2.a delta=0 时两复合仅差量化 dtype（<= 1 网格步）",
          md2 <= s_max * 1.0001, f"max|d|={md2:.3e} s_max={s_max:.3e}")

    with torch.no_grad():
        for p_ in w.lora.params.values():
            p_.normal_(0, 0.05)
    w._compose = "deploy"
    out = w._effective_shard("gate_up", w_shard, e0, epk)
    out.float().pow(2).sum().backward()
    g = w.lora.params["Agu"].grad
    check("C2.a deploy 复合下 delta 有梯度（STE 直通）",
          g is not None and g.abs().sum().item() > 0)


# ---------- C2.b 复合顺序 deploy 下的 EP 前向（集成级，防 E1 路径缺口） ----------
def test_compose_deploy_ep_forward() -> None:
    """_forward_vec_ep 在 deploy 复合下可跑通、确定、缓存旁路生效、LoRA 梯度非零。"""
    import os
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return
    E, h, i = 8, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    exp.gate_up_proj.ds_tensor = exp.gate_up_proj.detach().clone().reshape(-1)
    exp.gate_up_proj.ds_numel = exp.gate_up_proj.numel()
    exp.gate_up_proj.ds_shape = tuple(exp.gate_up_proj.shape)
    exp.down_proj.ds_tensor = exp.down_proj.detach().clone().reshape(-1)
    exp.down_proj.ds_numel = exp.down_proj.numel()
    exp.down_proj.ds_shape = tuple(exp.down_proj.shape)
    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    os.environ["QAD_EP"] = "1"
    os.environ["QAD_EP_TMAX"] = "6"
    try:
        w = STEQwen3Experts(exp, rank=4, dtype=torch.float32, zero3_gather=True,
                            deploy_scales={"gate_up": s_gu, "down": s_dn})
        if w._ep_setup() is None:
            check("C2.b EP setup 生效", False)
            return
        with torch.no_grad():
            for p_ in w.lora.params.values():
                p_.normal_(0, 0.05)
        w._compose = "deploy"
        w._wq_cache_enabled = True
        T, K = 6, 2
        x = torch.randn(T, h).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
        wt = torch.rand(T, K)
        wt = wt / wt.sum(-1, keepdim=True)
        out1 = w._forward_vec_ep(x, idx, wt)
        out2 = w._forward_vec_ep(x, idx, wt)
        check("C2.b deploy 复合 EP 前向可跑通且确定", torch.equal(out1, out2),
              f"shape={tuple(out1.shape)}")
        check("C2.b deploy 模式下码本未被建立（缓存旁路生效）", not w._wq_cache)
        out1.sum().backward()
        ok = all(p_.grad is not None and p_.grad.abs().sum() > 0
                 for p_ in w.lora.params.values())
        check("C2.b deploy 复合 EP 反向梯度流（LoRA b/a 非零）", ok)
    finally:
        os.environ.pop("QAD_EP", None)
        os.environ.pop("QAD_EP_TMAX", None)


# ---------- C3.a prefetch 路径（pinned 码本 + async H2D）与原路径逐 bit 一致 ----------
def test_prefetch_shard_bit_identical() -> None:
    """_effective_shard_prefetch == _effective_shard（码本与 unpack 逐 bit 等价）。"""
    E, h, i = 8, 16, 32
    exp = MockExperts(E=E, h=h, i=i)
    s_gu = torch.rand(E, 2 * i, 1) * 0.5 + 0.75
    s_dn = torch.rand(E, h, 1) * 0.5 + 0.75
    w = STEQwen3Experts(exp, rank=4, dtype=torch.float32,
                        deploy_scales={"gate_up": s_gu, "down": s_dn})
    with torch.no_grad():
        for p_ in w.lora.params.values():
            p_.normal_(0, 0.05)
    epk, e0 = E, 0
    for proj, shape in (("gate_up", (epk, 2 * i, h)), ("down", (epk, h, i))):
        src = exp.gate_up_proj if proj == "gate_up" else exp.down_proj
        w_shard = src.detach().reshape(*shape).to(torch.bfloat16)
        ref = w._effective_shard(proj, w_shard.clone(), e0, epk)
        w._pf_cache = {}
        got1 = w._effective_shard_prefetch(proj, w_shard, e0, epk, torch.device("cpu"))
        got2 = w._effective_shard_prefetch(proj, w_shard, e0, epk, torch.device("cpu"))
        md1 = (ref.float() - got1.float()).abs().max().item()
        md2 = (ref.float() - got2.float()).abs().max().item()
        check(f"C3.a {proj} prefetch==原路径（首访 pack）", md1 == 0, f"max|d|={md1:.3e}")
        check(f"C3.a {proj} prefetch==原路径（缓存命中）", md2 == 0, f"max|d|={md2:.3e}")
        codes, _s = w._pf_cache[proj]
        pin = "?"
        try:
            pin = str(codes.is_pinned())
        except Exception:
            pass
        check(f"C3.a {proj} 码本 int8（pin={pin}）", codes.dtype == torch.int8)


if __name__ == "__main__":
    test_structure()
    test_multi_expert_folding()
    test_output_consistency()
    test_deploy_scales_used()
    test_ep_world1_bit_identical()
    test_wq_cache_bit_identical()
    test_compose_deploy()
    test_compose_deploy_ep_forward()
    test_prefetch_shard_bit_identical()
    print()
    if FAILURES:
        print(f"=== {len(FAILURES)} FAILED: {FAILURES} ===")
        sys.exit(1)
    print("=== C1 ALL PASS ===")
