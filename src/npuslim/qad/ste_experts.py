"""Qwen3 MoE 专家层的 STE fake-quant + per-expert LoRA 集成（M4 训练核心构件）。

原理（§4.4 Phase 1 / M1）：
- 训练时学生有效权重 = fakequant(W) + LoRA delta（output-space，B 零初始化）
- fake-quant 精确复刻部署网格（对称 per-output-channel int4，见 ste.py）
- 包装方式：替换 Qwen3MoeExperts.forward，用有效权重复刻原计算，
  autograd 记录在有效权重上 → 梯度经 STE 直通回 LoRA 参数（W 冻结无梯度）

性能（方案 A 批处理改造，2026-09-04）：
- `_forward_vec`：全向量化（无 Python 专家循环）——
  gate_up 用一次 `einsum('td,ehd->teh')` 算全专家输出再 gather topk；
  down 用稀疏 scatter 后一次 `einsum('tei,eid->ted')`。
  代价是 128 专家全算（16x 计算冗余），但消除逐专家 kernel 启动开销，
  bf16 矩阵乘，吞吐较逐专家版提升一个量级以上。
- `_forward_loop`：保留原逐专家版（torch.where + linear + index_add_），
  用于向量版的数值对拍（两者应在浮点误差内一致）。
"""
from __future__ import annotations

import os

import torch
from torch import nn

from npuslim.qad.ste import fake_quant_weight
from npuslim.qad.lora import ExpertLoRA


_STE_PG = None   # STE 专用进程组（与引擎 ZeRO-3 gather 集合通信隔离）
_DBG_N = 0       # gather 对拍采样计数（前 4 次：1 次手动 + 前 3 次_forward 中）


def _get_ste_pg():
    """STE 手动 all_gather 专用进程组（2026-09-07 探针假设，调研文档 §10.5）。

    engine() forward 期间引擎在默认进程组的独立 NPU 流上做 ZeRO-3 参数
    gather；STE 的 all_gather 若与之同组并发，两侧集合通信交错：STE 接收
    缓冲收到引擎在途的其它层分片（错层），STE 的在途分片也污染引擎
    gather 结果（ds_tensor 内容错层的另一解释）。独立通信域隔离两侧。
    """
    global _STE_PG
    if _STE_PG is None:
        import torch.distributed as dist
        _STE_PG = dist.new_group()
    return _STE_PG


def _dbg_gather(tag, shard, parts, full):
    global _DBG_N
    _DBG_N += 1
    if _DBG_N > 4 or int(os.environ.get("RANK", 0)) != 0:
        return
    print(f"[gather-dbg#{_DBG_N}:{tag}] "
          f"shard_in[0:4]={shard[:4].float().flatten().tolist()} "
          f"parts0[0:4]={parts[0][:4].float().flatten().tolist()} "
          f"parts1[0:4]={parts[1][:4].float().flatten().tolist()} "
          f"full[0:4]={full.view(-1)[:4].float().flatten().tolist()}", flush=True)


def _gather_full_from_shard(shard, ds_numel, ds_shape) -> torch.Tensor:
    """从单 rank 分片 all_gather 重组完整参数（bf16）。

    分片沿扁平维度连续切分（ceil(ds_numel/world)），all_gather 拼接后截断
    padding 即可精确还原原权重（bf16↔fp16 数值无损）。
    """
    import torch.distributed as dist
    shard = shard.contiguous()
    world = dist.get_world_size()
    if world == 1:
        full = shard[:ds_numel]
    else:
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard, group=_get_ste_pg())
        full = torch.cat(parts, dim=0)[:ds_numel]
        if _DBG_N < 4:
            _dbg_gather(f"n{shard.numel()}", shard, parts, full)
    return full.view(list(ds_shape)).to(torch.bfloat16)


def _zero3_gather_full(p) -> torch.Tensor:
    """从 ZeRO-3 参数的 ds_tensor 手动 all_gather 重组完整参数（bf16）。

    注意（2026-09-07 checksum 探针实证，调研文档 §10.3）：engine() forward
    期间引擎会把 ds_tensor 重指向 gather 缓冲（data_ptr 变化、内容错层，
    forward 后恢复），本函数**仅在 forward 前/后可用**；forward 中必须使用
    initialize 后做的私有快照（STEQwen3Experts.snapshot_ds_tensors +
    _effective_all 快照分支）。
    """
    return _gather_full_from_shard(p.ds_tensor, p.ds_numel, p.ds_shape)


class STEQwen3Experts(nn.Module):
    """包装 Qwen3MoeExperts：STE fake-quant 权重 + per-expert LoRA delta。

    用法：`wrapped = STEQwen3Experts(original_experts, rank=64)` 后把
    `layer.mlp.experts` 替换为 wrapped（保留原模块引用用于权重冻结）。
    """

    def __init__(self, experts: nn.Module, rank: int = 64,
                 dtype: torch.dtype = torch.float32, vectorized: bool = True,
                 use_checkpoint: bool = False, zero3_gather: bool = False):
        super().__init__()
        self.experts = experts                      # 原 Qwen3MoeExperts（权重冻结）
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn
        self.vectorized = vectorized
        self.use_checkpoint = use_checkpoint
        self.zero3_gather = zero3_gather
        self.lora = ExpertLoRA(num_experts=self.num_experts,
                               hidden_size=self.hidden_dim,
                               moe_intermediate_size=self.intermediate_dim,
                               rank=rank, dtype=dtype)
        self._ds_snapshot: dict = {}   # proj -> ds_tensor 私有快照（方案 A，诊断用）
        self._ds_cpu: dict = {}        # proj -> ds_tensor CPU 快照（方案 A2，权威源）

    def _effective_all(self, proj: str) -> torch.Tensor:
        """全专家有效权重 = fakequant(W) + LoRA delta（[E, out, in] bf16）。

        每 forward 每 proj 一次。bf16 计算以减半瞬态显存（FSDP 下 fp32 的
        3D 权重 + 逐层 autograd 保存会撑爆 64GB）；int4 网格粗，bf16 量化
        舍入对训练影响可接受。

        DeepSpeed ZeRO-3 注：引擎 hook gather 在本场景写入错层缓冲，且
        checksum 探针（2026-09-07）实证 forward 期间 ds_tensor 本身也被引擎
        重指向（data_ptr 变化、内容错层）。故 zero3_gather=True 时优先读
        initialize 后、首次 forward 前的私有快照（snapshot_ds_tensors）；
        无快照时回退 ds_tensor 现场重组（仅 forward 前/后正确）。
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
                # 方案 A2：NPU 私有快照存储也会被引擎 forward 改写（2026-09-07
                # 探针：snap_gu 创建时==file、forward 中变错层），故每次从 CPU
                # 快照重新上传分片再 all_gather——H2D 全量覆写目标缓冲，引擎
                # 触不可及 CPU 源。验证期用同步 H2D（non_blocking=False），
                # 排除跨流异步风险；性能优化待正确性闭环后再做
                shard = cpu_snap.to(w.device, non_blocking=False)
                w_full = _gather_full_from_shard(shard, w.ds_numel, w.ds_shape)
            elif snap is not None:
                w_full = _gather_full_from_shard(snap, w.ds_numel, w.ds_shape)
            elif self.zero3_gather and getattr(w, "ds_tensor", None) is not None:
                w_full = _zero3_gather_full(w)     # 回退路径，见函数 docstring
            else:
                w_full = w.detach().to(torch.bfloat16)
            wq = fake_quant_weight(w_full, bits=4)  # bf16 网格
        d = torch.einsum("eor,eri->eoi", b, a) * self.lora.scaling
        return (wq + d.to(torch.bfloat16)).to(torch.bfloat16)

    def snapshot_ds_tensors(self):
        """engine.initialize 后、首次 forward 前调用：clone 专家参数 ds_tensor。

        forward 期间 ds_tensor 被引擎重指向（调研文档 §10.3 checksum 探针
        data_ptr 实证）；且 NPU 私有快照的存储同样会被 forward 改写（方案 A
        探针：clone 时内容==file、forward 中变错层）。故方案 A2 同时保留：
        - _ds_cpu：CPU 权威快照（引擎触不可及），_effective_all 每次从它
          重新 H2D 上传分片（每 rank 主存 ≈ 专家权重总量/world，Qwen3-30B
          8 卡 ≈9.2GB；每次 forward 每 proj +96MB H2D）
        - _ds_snapshot：NPU 快照，仅作诊断对拍（验证引擎改写行为）
        """
        for proj, p in (("gate_up", self.experts.gate_up_proj),
                        ("down", self.experts.down_proj)):
            dt = getattr(p, "ds_tensor", None)
            if dt is not None:
                self._ds_snapshot[proj] = dt.detach().clone()
                cpu = dt.detach().to("cpu", copy=True)
                try:
                    cpu = cpu.pin_memory()
                except Exception:
                    pass                    # pinned 不可用时退化为 pageable
                self._ds_cpu[proj] = cpu

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        if self.vectorized:
            if self.use_checkpoint:
                # 反向重算 STE 有效权重与中间量，避免 48 层 × fp32 权重被 autograd
                # 保存导致 OOM（FSDP 场景必须开）
                return torch.utils.checkpoint.checkpoint(
                    self._forward_vec, hidden_states, top_k_index, top_k_weights,
                    use_reentrant=False)
            return self._forward_vec(hidden_states, top_k_index, top_k_weights)
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)

    def _forward_vec(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        """全向量化批处理：无专家循环，autograd 完整。"""
        T = hidden_states.shape[0]
        K = top_k_index.shape[1]
        i_dim, h_dim = self.intermediate_dim, self.hidden_dim
        E = self.num_experts

        w_gu = self._effective_all("gate_up")      # [E, 2i, h] fp32
        w_dn = self._effective_all("down")         # [E, h, i] fp32
        x = hidden_states.to(torch.bfloat16)
        w_gu_b = w_gu.to(torch.bfloat16)
        w_dn_b = w_dn.to(torch.bfloat16)

        # gate_up：原生 matmul 一次算全专家 [T,E,2i]（einsum 在 NPU 编译慢，用 matmul）
        #   x [T,h] @ W^T [h, E*2i] → [T, E*2i] → reshape [T,E,2i]
        gu_all = torch.matmul(x, w_gu_b.reshape(E * 2 * i_dim, h_dim).transpose(0, 1))
        gu_all = gu_all.view(T, E, 2 * i_dim)
        idx_gu = top_k_index.unsqueeze(-1).expand(T, K, 2 * i_dim)
        gate_up = gu_all.gather(1, idx_gu)         # [T,K,2i]
        gate, up = gate_up.chunk(2, dim=-1)
        h = self.act_fn(gate) * up                 # [T,K,i] bf16

        # down：稀疏 h → 原生 bmm 全专家 [T,E,h] → gather topk → 路由加权求和
        #   h_sparse [T,E,i] → permute [E,T,i] @ w_dn [E,i,h] → [E,T,h] → [T,E,h]
        h_sparse = torch.zeros(T, E, i_dim, dtype=torch.bfloat16, device=hidden_states.device)
        h_sparse.scatter_(1, top_k_index.unsqueeze(-1).expand(T, K, i_dim), h)
        down_all = torch.bmm(h_sparse.permute(1, 0, 2),
                             w_dn_b.permute(0, 2, 1)).permute(1, 0, 2)
        idx_dn = top_k_index.unsqueeze(-1).expand(T, K, h_dim)
        out = down_all.gather(1, idx_dn)           # [T,K,h]
        out = out * top_k_weights.unsqueeze(-1)
        return out.sum(dim=1).to(hidden_states.dtype)

    def _forward_loop(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        """原逐专家版（保留作数值对拍基线）。"""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        w_gu = self._effective_all("gate_up")
        w_dn = self._effective_all("down")
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])          # Python int：w_gu[int] → 2D
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, w_gu[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, w_dn[expert_idx])
            current_hidden_states = current_hidden_states * \
                top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def wrap_all_experts(model, rank: int = 64, vectorized: bool = True,
                     dtype: torch.dtype = torch.float32,
                     use_checkpoint: bool = False,
                     zero3_gather: bool = False) -> list[STEQwen3Experts]:
    """给 Qwen3MoE 模型的全部专家层套 STE+LoRA 包装，返回包装列表。

    冻结原专家权重（W 不参与训练），LoRA 参数可训练。
    dtype: LoRA 参数精度（FSDP 扁平化要求与模型权重同 dtype，用 bf16）。
    use_checkpoint: gradient checkpointing（FSDP 场景防 OOM 必须开）。
    zero3_gather: DeepSpeed ZeRO-3 场景开启（见 _effective_all）。
    """
    wrapped = []
    for layer in model.model.layers:
        exp = layer.mlp.experts
        for p in exp.parameters():
            p.requires_grad = False
        w = STEQwen3Experts(exp, rank=rank, vectorized=vectorized, dtype=dtype,
                            use_checkpoint=use_checkpoint, zero3_gather=zero3_gather)
        w.to(exp.gate_up_proj.device)          # LoRA 参数移到层所在设备（device_map 分片）
        layer.mlp.experts = w
        wrapped.append(w)
    return wrapped
