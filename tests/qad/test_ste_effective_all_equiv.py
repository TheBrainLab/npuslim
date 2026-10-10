"""数值等价测试：NEW 逐专家 slab 流式 _effective_all vs OLD 全张量参考实现。

S2 迁移（§6e-5）把 _effective_all 改为逐专家 slab 流式以消除 ~51.6 GB fp32
瞬态。本测试把迁移前的 OLD _effective_all 逐字保留为参考函数，对以下配置
矩阵断言 NEW 与 OLD 逐 bit 一致（forward maxdiff == 0）且对 LoRA b,a 的
梯度完全相同：
  - deploy_scales: present / absent(minmax 回退)
  - trainable scale: on(QAD_TRAIN_SCALES=1) / off
  - zero3_gather: on(ds_tensor 分支，_zero3_gather_full 打桩) / off(w.detach 回退)

可导性核对：OLD/NEW 均把 wq 构造在 no_grad 内（scale 梯度按历史不回传），
d 构造在 no_grad 外（梯度经 d → b,a）。故 trainable 时 scale.grad 两边均
为 None，b/a.grad 两边一致——锁定"训练语义不变"。

运行（容器内）：
  pytest tests/qad/test_ste_effective_all_equiv.py -q
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from npuslim.qad import ste_experts as _ste_mod
from npuslim.qad.ste_experts import STEQwen3Experts
from npuslim.qad.ste import compute_weight_scale, fake_quant_weight


def _effective_all_old(self, proj: str) -> torch.Tensor:
    """迁移前 OLD _effective_all 的逐字副本（等价参考）。

    自管快照分支保留；_ds_cpu/_ds_snapshot 在新实现里为空 dict，故 cpu_snap/
    snap 恒为 None → 落到与 NEW 相同的 LIVE ds_tensor / w.detach 源，使对比
    只隔离"全张量 vs 逐 slab"的计算重排。
    """
    if proj == "gate_up":
        w = self.experts.gate_up_proj
        b, a = self.lora.params["Bgu"], self.lora.params["Agu"]
    else:
        w = self.experts.down_proj
        b, a = self.lora.params["Bdn"], self.lora.params["Adn"]
    with torch.no_grad():
        cpu_snap = self._ds_cpu.get(proj) if self.zero3_gather else None
        snap = self._ds_snapshot.get(proj) if self.zero3_gather else None
        if cpu_snap is not None:
            from npuslim.qad.ste_experts import _gather_full_from_shard
            shard = cpu_snap.to(w.device, non_blocking=False)
            w_full = _gather_full_from_shard(shard, w.ds_numel, w.ds_shape)
        elif snap is not None:
            from npuslim.qad.ste_experts import _gather_full_from_shard
            w_full = _gather_full_from_shard(snap, w.ds_numel, w.ds_shape)
        elif self.zero3_gather and getattr(w, "ds_tensor", None) is not None:
            w_full = _ste_mod._zero3_gather_full(w)
        else:
            w_full = w.detach().to(torch.bfloat16)
        scale = None
        scale_trainable = proj in self.scale_params
        if scale_trainable:
            scale = self.scale_params[proj]
        elif self.deploy_scales is not None:
            scale = self.deploy_scales.get(proj)
            if scale is not None:
                scale = scale.to(w_full.device)
        if scale is None:
            scale = compute_weight_scale(w_full, bits=4).to(w_full.device)
        with torch.no_grad():
            v = torch.clamp(torch.round(w_full / scale), -8, 7)
        if scale_trainable:
            wq = scale.to(w_full.device) * v
        else:
            wq = fake_quant_weight(w_full, bits=4, scale=scale)
    d = torch.einsum("eor,eri->eoi", b, a) * self.lora.scaling
    return (wq + d.to(torch.bfloat16)).to(torch.bfloat16)


class MockExperts(nn.Module):
    """模拟 GlmMoeDsaNaiveMoe（与 Qwen3MoeExperts 同构）；fp32 权重。"""

    def __init__(self, E: int, h: int, i: int):
        super().__init__()
        self.num_experts = E
        self.hidden_dim = h
        self.intermediate_dim = i
        self.act_fn = nn.SiLU()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * i, h) * 0.3)
        self.down_proj = nn.Parameter(torch.randn(E, h, i) * 0.3)


def _build_pair(E, h, i, rank, deploy_scales, zero3_gather, trainable, with_ds_tensor):
    """构造共享冻结专家、LoRA 同初始化的 NEW/OLD 一对 wrapper。"""
    torch.manual_seed(123)
    exp = MockExperts(E=E, h=h, i=i)
    if with_ds_tensor:
        # 挂一个占位 ds_tensor，使 NEW/OLD 都走 _zero3_gather_full 分支（由打桩返回 w.detach().to(bf16)）
        exp.gate_up_proj.ds_tensor = torch.empty(1, dtype=torch.bfloat16)
        exp.gate_up_proj.ds_numel = exp.gate_up_proj.numel()
        exp.gate_up_proj.ds_shape = tuple(exp.gate_up_proj.shape)
        exp.down_proj.ds_tensor = torch.empty(1, dtype=torch.bfloat16)
        exp.down_proj.ds_numel = exp.down_proj.numel()
        exp.down_proj.ds_shape = tuple(exp.down_proj.shape)
    if trainable:
        os.environ["QAD_TRAIN_SCALES"] = "1"
    else:
        os.environ.pop("QAD_TRAIN_SCALES", None)
    torch.manual_seed(456)
    w_new = STEQwen3Experts(exp, rank=rank, dtype=torch.float32,
                            zero3_gather=zero3_gather, deploy_scales=deploy_scales)
    torch.manual_seed(456)
    w_old = STEQwen3Experts(exp, rank=rank, dtype=torch.float32,
                            zero3_gather=zero3_gather, deploy_scales=deploy_scales)
    # 同种子随机化 LoRA B/A（B 非零 → d 非平凡，梯度有意义）
    for w in (w_new, w_old):
        torch.manual_seed(789)
        with torch.no_grad():
            for p in w.lora.params.values():
                p.normal_(0, 0.05)
    return exp, w_new, w_old


# (deploy_scales, trainable, zero3_gather, with_ds_tensor)
_CASES = [
    (None,     False, False, False),   # minmax, w.detach
    (None,     False, True,  True),    # minmax, gather 分支(打桩)
    ("deploy", False, False, False),   # deploy scale, w.detach
    ("deploy", False, True,  True),    # deploy scale, gather
    ("deploy", True,  False, False),   # trainable scale, w.detach
    ("deploy", True,  True,  True),    # trainable scale, gather
]


def _make_deploy(E, h, i):
    return {
        "gate_up": (torch.rand(E, 2 * i, 1) * 0.5 + 0.75),
        "down": (torch.rand(E, h, 1) * 0.5 + 0.75),
    }


class TestEffectiveAllEquiv:
    # E=10 与 SLAB=8 → slabs [0:8],[8:10]，覆盖非整除边界
    E, H, I, RANK = 10, 8, 16, 4

    def _run_case(self, deploy_kind, trainable, zero3_gather, with_ds_tensor, proj):
        deploy = _make_deploy(self.E, self.H, self.I) if deploy_kind == "deploy" else None
        orig_gather = _ste_mod._zero3_gather_full
        if with_ds_tensor:
            # 打桩 gather：返回该参数的 bf16 全量（模拟 all_gather 重组）
            _ste_mod._zero3_gather_full = lambda p: p.detach().to(torch.bfloat16)
        try:
            _, w_new, w_old = _build_pair(self.E, self.H, self.I, self.RANK,
                                          deploy, zero3_gather, trainable, with_ds_tensor)
            # forward 等价
            new = w_new._effective_all(proj)
            old = _effective_all_old(w_old, proj)
        finally:
            _ste_mod._zero3_gather_full = orig_gather
        md = (new.float() - old.float()).abs().max().item()
        assert md == 0.0, (
            f"[{proj} deploy={deploy_kind} trainable={trainable} "
            f"z3={zero3_gather} ds={with_ds_tensor}] forward maxdiff={md} (expect 0)")

        # 梯度等价：重新建一对（清零 grad），各自 backward，比对 b,a 梯度
        orig_gather = _ste_mod._zero3_gather_full
        if with_ds_tensor:
            _ste_mod._zero3_gather_full = lambda p: p.detach().to(torch.bfloat16)
        try:
            _, w_n, w_o = _build_pair(self.E, self.H, self.I, self.RANK,
                                      deploy, zero3_gather, trainable, with_ds_tensor)
            out_n = w_n._effective_all(proj)
            out_o = _effective_all_old(w_o, proj)
        finally:
            _ste_mod._zero3_gather_full = orig_gather
        out_n.sum().backward()
        out_o.sum().backward()
        for pname in ("Bgu", "Agu") if proj == "gate_up" else ("Bdn", "Adn"):
            gn = w_n.lora.params[pname].grad
            go = w_o.lora.params[pname].grad
            assert gn is not None and go is not None, (
                f"[{pname} {proj}] grad is None (NEW={gn is None} OLD={go is None})")
            gmd = (gn.float() - go.float()).abs().max().item()
            assert gmd == 0.0, (
                f"[{pname} {proj} deploy={deploy_kind} trainable={trainable} "
                f"z3={zero3_gather}] grad maxdiff={gmd} (expect 0)")
        # trainable scale：wq 在 no_grad 内 → scale 梯度两边均为 None（训练语义不变）
        if trainable:
            sn = w_n.scale_params[proj].grad
            so = w_o.scale_params[proj].grad
            assert sn is None and so is None, (
                f"[{proj} trainable] scale.grad NEW={sn} OLD={so} (expect None,None)")

    def test_equiv_gate_up_all_configs(self):
        for deploy_kind, trainable, z3, ds in _CASES:
            self._run_case(deploy_kind, trainable, z3, ds, "gate_up")

    def test_equiv_down_all_configs(self):
        for deploy_kind, trainable, z3, ds in _CASES:
            self._run_case(deploy_kind, trainable, z3, ds, "down")

    def test_snapshot_is_noop_and_attrs_empty(self):
        """回归：S2 迁移后 snapshot_ds_tensors 为 no-op，_ds_cpu/_ds_snapshot 恒空。"""
        exp = MockExperts(self.E, self.H, self.I)
        os.environ.pop("QAD_TRAIN_SCALES", None)
        w = STEQwen3Experts(exp, rank=self.RANK, dtype=torch.float32,
                            zero3_gather=True, deploy_scales=_make_deploy(self.E, self.H, self.I))
        w.snapshot_ds_tensors()
        assert w._ds_snapshot == {}
        assert w._ds_cpu == {}
