"""Tiny MoE 模型：QAD 训练通路（M3 sanity）与 A7 消融共用的 CPU 级验证模型。

结构同构 GLM-5.2 MoE 专家块：fused gate_up (2*inter) + swiglu + down，top-1 路由。
刻意保持极小（CPU 秒级前向），但覆盖 QAD 训练全部环节：
- 专家权重 STE fake-quant（weight_provider 注入有效权重，梯度只走 LoRA）
- per-expert LoRA 挂载（见 lora.ExpertLoRA）
- chunked KL 蒸馏（见 kl_loss.kl_distill_loss）

无 HF/transformers 依赖，避免版本耦合；仅用于数值与通路验证，非部署产物。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .ste import fake_quant_weight


@dataclass
class TinyMoEConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    inter_size: int = 64          # 专家 intermediate（gate_up out = 2*inter）
    num_experts: int = 4
    num_layers: int = 2
    num_heads: int = 2


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def fake_quant_act_per_token(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """per-token 动态 fake-quant（W4A8 部署语义，A7 eval 期模拟）。

    离群通道会把整 token 的 scale 拉大、压低其余通道的量化精度——这正是
    旋转/AWQ 要缓解的误差来源，也是 tiny 消融必须有激活量化的原因
    （纯 W4-only 下旋转对权重误差中性，消融无分辨力，2026-09-02 实测）。
    tiny 上 int8（127 级）误差远低于 ppl 分辨力，harness 用 bits=4 作
    压力测试放大镜（非部署语义，用于测相对收益排序）。
    """
    qmax = 2 ** (bits - 1) - 1
    s = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return (x.float() / s).round().clamp(-qmax - 1, qmax).mul_(s).to(x.dtype)


class ExpertMLP(nn.Module):
    """单个专家：gate_up fused + swiglu + down。weight_provider 注入训练期有效权重。

    forward 额外接受 r2（A7 旋转：down 输入边界白化），并把最近一次前向的
    输入 / down 输入（均已处于旋转后基）stash 到 _last_in/_last_h，供
    A7 harness 采集 AWQ 统计（x_abs_mean 必须基于折叠后权重产生的激活）。
    """

    def __init__(self, cfg: TinyMoEConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.gate_up = nn.Parameter(torch.empty(2 * cfg.inter_size, cfg.hidden_size))
        self.down = nn.Parameter(torch.empty(cfg.hidden_size, cfg.inter_size))
        self.reset_parameters()
        # QAD 训练期由外部注入 fn(expert) -> (gate_up_w, down_w)；None 走原始 float 权重
        self.weight_provider = None
        self.activation_quant = False  # A7: eval 期 per-token 激活量化模拟（W4A8 语义）
        self.act_bits = 8
        self._last_in: torch.Tensor | None = None
        self._last_h: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        for p in (self.gate_up, self.down):
            nn.init.normal_(p, std=0.02)

    def forward(self, x: torch.Tensor, r2: torch.Tensor | None = None) -> torch.Tensor:
        if self.weight_provider is not None:
            w_gu, w_dn = self.weight_provider(self)
        else:
            w_gu, w_dn = self.gate_up, self.down
        self._last_in = x.detach()  # 统计语义：量化器输入的原始激活（旋转后基）
        if self.activation_quant:
            x = fake_quant_act_per_token(x, self.act_bits)
        gu = x @ w_gu.T
        g, u = gu.chunk(2, dim=-1)
        h = F.silu(g) * u
        if r2 is not None:
            h = h @ r2.T
        self._last_h = h.detach()
        if self.activation_quant:
            h = fake_quant_act_per_token(h, self.act_bits)
        return h @ w_dn.T


class MoELayer(nn.Module):
    """top-1 路由 MoE（router 冻结，训练梯度只经专家有效权重回传 LoRA）。"""

    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            ExpertMLP(cfg, expert_id=i) for i in range(cfg.num_experts)
        )

    def forward(self, x: torch.Tensor, r2: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, H]
        b, l, h = x.shape
        flat = x.reshape(-1, h)
        idx = self.router(flat).argmax(dim=-1)                       # [B*L] top-1
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = idx == e
            if mask.any():
                out[mask] = expert(flat[mask], r2)
        return out.reshape(b, l, h)


class Attention(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, l, self.n_heads, self.head_dim)
        q, k, v = (t.reshape(shape).transpose(1, 2) for t in (q, k, v))  # [B, nh, L, hd]
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        causal = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, l, h)
        return self.proj(out)


class MoEBlock(nn.Module):
    """Attention + MoE 块。

    A7 旋转注入点（r1/r2 由 apply_rotation 设置，默认 None 走原始前向）：
    - r1 在 RMSNorm **之后**注入（RMSNorm 的 gain g 不满足旋转不变性，
      (Rx)⊙g ≠ R(x⊙g)，norm 前旋转会破坏等价性——2026-09-02 数学裁定）
    - 注入点为量化 matmul 边界：qkv / router / gate_up 输入（r1）、down 输入（r2），
      与部署在线 kernel 的作用位置一致（§6.1 Step 3）
    """

    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.moe_norm = RMSNorm(cfg.hidden_size)
        self.moe = MoELayer(cfg)
        self.r1: torch.Tensor | None = None
        self.r2: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = self.attn_norm(x) if self.r1 is None else self.attn_norm(x) @ self.r1.T
        x = x + self.attn(x_attn)
        x_moe = self.moe_norm(x) if self.r1 is None else self.moe_norm(x) @ self.r1.T
        return x + self.moe(x_moe, self.r2)


class TinyMoEModel(nn.Module):
    """tiny causal LM：embed + N x MoEBlock + norm + lm_head。"""

    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg.num_layers))
        # expert_id 跨层全局唯一（LoRA delta 按此索引，避免跨层共享）
        for eid, expert in enumerate(self.iter_experts()):
            expert.expert_id = eid
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L] -> logits: [B, L, V]
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def iter_experts(self):
        for block in self.blocks:
            yield from block.moe.experts


def attach_qad_lora(model: TinyMoEModel, lora, bits: int = 4) -> None:
    """QAD 训练态挂载：专家前向改用 fakequant(W) + LoRA delta 的有效权重。

    fake_quant_weight 的输入是冻结的原始权重（requires_grad=False），
    delta 携带梯度 —— 梯度经 STE 直通到原始权重图但不产生参数更新，
    只有 LoRA 参数可训练（调用方 optimizer 只收 lora.trainable_parameters()）。
    """
    for expert in model.iter_experts():
        e = expert.expert_id

        def provider(exp: ExpertMLP, _e: int = e) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                fake_quant_weight(exp.gate_up, bits) + lora.delta("gate_up", _e),
                fake_quant_weight(exp.down, bits) + lora.delta("down", _e),
            )

        expert.weight_provider = provider


def detach_qad_lora(model: TinyMoEModel) -> None:
    for expert in model.iter_experts():
        expert.weight_provider = None


def apply_rotation(model: TinyMoEModel, r1: torch.Tensor | None,
                   r2s: list[torch.Tensor | None]) -> None:
    """A7 量化期旋转：权重侧折叠 + hook 注入（同时生效，二者缺一即静默错误）。

    折叠（行向量约定，y = x @ W.T）：
    - 消费 r1 旋转激活的权重列折 W ← W @ r1.T：qkv / router / gate_up
      （router 消费同一层输入，调研文档 §6.1 未单列，折叠为必要补充）
    - 消费 r2 旋转激活的 down 输入折 W ← W @ r2.T
    - o_proj 不折：其输入（attention 内部激活）未被旋转（A7 只在两处
      量化边界注入，与调研文档 hook 方案一致）

    Args:
        r1: [H, H] 共享旋转阵；None 表示不启用 R1
        r2s: 每层 [I, I] 旋转阵列表；元素 None 表示该层不启用 R2
    """
    assert len(r2s) == len(model.blocks)
    for block, r2 in zip(model.blocks, r2s):
        with torch.no_grad():
            if r1 is not None:
                block.attn.qkv.weight.data = block.attn.qkv.weight.data @ r1.T
                block.moe.router.weight.data = block.moe.router.weight.data @ r1.T
                for e in block.moe.experts:
                    e.gate_up.data = e.gate_up.data @ r1.T
            if r2 is not None:
                for e in block.moe.experts:
                    e.down.data = e.down.data @ r2.T
        block.r1 = r1
        block.r2 = r2


def clear_rotation(model: TinyMoEModel) -> None:
    for block in model.blocks:
        block.r1 = None
        block.r2 = None
