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

from npuslim.qad.ste import compute_weight_scale, fake_quant_weight
from npuslim.qad.lora import ExpertLoRA


_STE_PG = None   # STE 专用进程组（与引擎 ZeRO-3 gather 集合通信隔离）
_STE_PG_HCCL = None  # NPU 分片用 HCCL 组（OFFLOAD_PARAM=0 路径）
_DBG_N = 0       # gather 对拍采样计数（前 4 次：1 次手动 + 前 3 次_forward 中）
_SLAB = int(os.environ.get("QAD_SLAB", "8"))   # 0927 A2：slab 粒度可调（8→2 细化分配颗粒，逐 bit 等价）


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
        # gloo 后端：STE 的 all_gather 作用于 CPU 分片（remote_device=cpu 时），
        # HCCL 对 CPU 张量在 4 节点下 SIGSEGV（09-16 C4-32 实测）；CPU 分片聚合走 gloo。
        _STE_PG = dist.new_group(backend="gloo")
    return _STE_PG


def _get_ste_pg_hccl():
    """NPU 分片用 HCCL 组（OFFLOAD_PARAM=0 路径）。

    与 _get_ste_pg 对称：OFFLOAD_PARAM=0 时 ds_tensor 在 NPU，gloo 无法处理
    NPU 张量；用独立 HCCL 组做 all_gather（隔离理由同 _get_ste_pg）。
    """
    global _STE_PG_HCCL
    if _STE_PG_HCCL is None:
        import torch.distributed as dist
        _STE_PG_HCCL = dist.new_group(backend="hccl")
    return _STE_PG_HCCL


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

    后端选择（Phase-2 2b）：shard 在 NPU 时用 all_gather_into_tensor 在默认
    HCCL 组上聚合（单一连续缓冲，比 all_gather 的 32 个独立张量 + cat 省 2×
    HBM；S2 retest 已证 engine forward 期独立通信域非必需，默认组安全）。
    shard 在 CPU 时走 gloo 独立组（OFFLOAD_PARAM=1 且未做 H2D 预搬的旧路径）。
    """
    import torch.distributed as dist
    shard = shard.contiguous()
    if not (dist.is_available() and dist.is_initialized()):
        world = 1
    else:
        world = dist.get_world_size()
    if world == 1:
        full = shard[:ds_numel]
    else:
        if shard.device.type == "npu":
            output = torch.empty(world * shard.numel(), dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(output, shard, group=_get_ste_pg_hccl())
            full = output[:ds_numel]
        else:
            parts = [torch.empty_like(shard) for _ in range(world)]
            dist.all_gather(parts, shard, group=_get_ste_pg())
            full = torch.cat(parts, dim=0)[:ds_numel]
        if _DBG_N < 4:
            _dbg_gather(f"n{shard.numel()}", shard, [shard, shard], full)
    return full.view(list(ds_shape)).to(torch.bfloat16)


def _zero3_gather_full(p) -> torch.Tensor:
    """从 ZeRO-3 参数的 ds_tensor 手动 all_gather 重组完整参数（bf16）。

    S2 retest 结论（§6e-5 / 调研文档 §10.5 反转）：历史“engine forward 期间
    ds_tensor 被重指向/内容错层”的假说已被证伪——真实根因是 _make_ckpt_fwd
    的闭包迟绑定 bug（已修复）。live engine-managed ds_tensor 在 forward 期间
    数值正确（Q1 无 corruption、Q2 与 safetensors maxdiff=0 bit-exact）。
    故 _effective_all 直接读本函数（S2 路径），不再依赖自管快照。

    性能优化（Phase-2 2b）：ds_tensor 在 CPU 时（OFFLOAD_PARAM=1），先将单
    rank 分片 H2D 到 NPU，再用 HCCL all_gather（NPU→NPU，快），替代旧的
    gloo all_gather（CPU→CPU，4 节点下极慢）+ 大块 H2D（全量 12.9GB）。
    新路径 H2D 仅 ~403MB（单 rank 分片）+ HCCL all_gather（NPU 互连带宽）。
    """
    shard = p.ds_tensor
    if shard.device.type == "cpu" and p.device.type == "npu":
        shard = shard.to(p.device)
    return _gather_full_from_shard(shard, p.ds_numel, p.ds_shape)


def _ep_t_max(T: int, device) -> int:
    """EP 交换的统一 token 容量。优先取 QAD_EP_TMAX（=micro-tokens，训练脚本
    启动时设置；T ≤ micro_tokens 恒成立）——零同步。env 未设时回退 all_reduce(MAX)
    （正确但每次调用一次 host 同步，仅离线/调试用）。"""
    v = os.environ.get("QAD_EP_TMAX")
    if v:
        return int(v)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t_t = torch.tensor([T], dtype=torch.int64, device=device)
        dist.all_reduce(t_t, op=dist.ReduceOp.MAX, group=_get_ste_pg_hccl())
        return int(t_t.item())
    return T


class _EpAllGatherTokens(torch.autograd.Function):
    """token 隐状态 all_gather（EP 第 1 跳）。

    forward: x_pad [t_max,H] → [world*t_max,H]（rank 块序）。
    backward: 无通信——本 rank 的 grad 即自己那块行切片。
    """

    @staticmethod
    def forward(ctx, x_pad: torch.Tensor, t_max: int) -> torch.Tensor:
        import torch.distributed as dist
        ctx.t_max = t_max
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            ctx.world = 1
            return x_pad
        ctx.world = dist.get_world_size()
        ctx.rank = dist.get_rank()
        flat = x_pad.contiguous().reshape(-1)
        out = torch.empty(ctx.world * flat.numel(), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(out, flat, group=_get_ste_pg_hccl())
        return out.view(ctx.world * t_max, x_pad.shape[1])

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.world == 1:
            return grad_out, None
        g = grad_out[ctx.rank * ctx.t_max:(ctx.rank + 1) * ctx.t_max]
        return g.contiguous(), None


class _EpAllToAllOutputs(torch.autograd.Function):
    """专家输出 all_to_all（EP 第 2 跳，等分）。

    forward: y_all [world*t_max, epk, D]（chunk i = 本 rank 专家对 rank i tokens
    的输出）→ 交换 → [t_max, E, D]（专家主序 e = j*epk + k）。
    backward: 反向同构 all_to_all——grad_out 按 expert 块切 world 份发回各
    owner rank，收回本 rank 专家对全部 tokens 的 grad（[world*t_max, epk, D]）。

    必须自定义 Function：集合通信不在 autograd 图内传播梯度，checkpoint 重算
    （grad 开启）时图在通信处断裂 → LoRA 参数拿不到梯度 → DeepSpeed IPG
    bucket 撞 param.grad=None（09-18 world=32 实测）。
    """

    @staticmethod
    def forward(ctx, y_all: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            ctx.world = 1
            return y_all
        ctx.world = dist.get_world_size()
        ctx.rank = dist.get_rank()
        N, epk, D = y_all.shape
        ctx.t_max = N // ctx.world
        ctx.epk, ctx.D = epk, D
        flat = y_all.contiguous().reshape(-1)
        out = torch.empty_like(flat)
        dist.all_to_all_single(out, flat, group=_get_ste_pg_hccl())
        return out.view(ctx.world, ctx.t_max, epk, D).permute(1, 0, 2, 3) \
                  .reshape(ctx.t_max, ctx.world * epk, D)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.world == 1:
            return grad_out
        g_in = grad_out.reshape(ctx.t_max, ctx.world, ctx.epk, ctx.D) \
                       .permute(1, 0, 2, 3).contiguous()
        flat = g_in.reshape(-1)
        out = torch.empty_like(flat)
        import torch.distributed as dist
        dist.all_to_all_single(out, flat, group=_get_ste_pg_hccl())
        return out.view(ctx.world * ctx.t_max, ctx.epk, ctx.D)


def _ep_dispatch_meta(topk_idx, epk, world, dev):
    """派发元数据（ord_ + counts），挂在 topk_idx 张量属性上（0923 泄漏修复）。

    重算缓存动机不变：checkpoint 重算以同一 topk_idx 对象再次进入 forward，
    命中缓存可免 tolist 的 2 次 host 同步（重算期同步排队等 DeepSpeed 梯度
    reduce，实测 ~3s/次 → 450s/step）。

    0923 修复：原模块级 dict 按 id 缓存、>256 整表清空——75 条/步跨步累积使
    清空落在 step-4 的 fwd 中段，被清层的当步重算 MISS（31 层 × 2 同步 × 3s
    ≈ +186s，smoke 实测 bwd 225→447s/步泄漏）。改为张量属性：元数据随对象
    存活（fwd→重算同对象天然命中；跨步新对象天然失效），无全局状态、无清空
    时点、无跨步垃圾。
    """
    import torch.distributed as dist
    hit = getattr(topk_idx, "_qad_ep_meta", None)
    if hit is not None:
        return hit
    P = topk_idx.numel()
    dst = topk_idx.reshape(-1) // epk
    ord_ = torch.argsort(dst)
    send_counts_t = torch.zeros(world, dtype=torch.int64, device=dev)
    send_counts_t.scatter_add_(0, dst, torch.ones_like(dst))
    if world == 1:
        send_counts, recv_counts = [P], [P]
    else:
        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t.contiguous(),
                               group=_get_ste_pg_hccl())
        send_counts = send_counts_t.tolist()
        recv_counts = recv_counts_t.tolist()
    meta = (ord_, send_counts, recv_counts)
    topk_idx._qad_ep_meta = meta
    return meta


_PF_STAGING: dict = {}   # (proj, device) -> 持久 bf16 暂存，跨层共享 ~500MB（0923：消灭 prefetch 每步 ~225GB 临时分配）


class _EpExpertBlock(torch.autograd.Function):
    """单片专家块（sparse dispatch 版，QAD_EP=1 的最终形态）。

    forward: 按 (token,slot) 对的目的 rank 排序 → all_to_all 派发隐状态与
    专家 id（每对 ~12KB，比 dense-y 方案的 1.6GB 小 128×）→ owner rank 用
    本地 8 专家有效权重冗余计算（gu_all/y_all 全专家 × 选中）→ gather 选中
    对 → all_to_all 返回 → 加权求和。
    backward: 反向同构 all_to_all ×2 + 全 dense matmul 梯度（grad_W 直接
    bmm，无分段循环）；save 仅 ~120MB/层（x2/gate/up/h/索引），checkpoint
    重算代价趋零。

    集合通信全部经自定义 Function（通信不在 autograd 图内传播，checkpoint
    重算时图断裂 → LoRA 无梯度 → DeepSpeed IPG param.grad=None，09-18 实测）。
    """

    @staticmethod
    def forward(ctx, hidden, topk_idx, topk_w, w_gu_eff, w_dn_eff, act_fn, epk):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            world, rank = 1, 0
        else:
            world, rank = dist.get_world_size(), dist.get_rank()
        T, K = topk_idx.shape
        H = hidden.shape[1]
        i_dim = w_dn_eff.shape[2]
        P = T * K
        dev = hidden.device
        x = hidden.to(torch.bfloat16)

        flat_e = topk_idx.reshape(-1)                       # [P] 全局专家 id
        ord_, send_counts, recv_counts = _ep_dispatch_meta(topk_idx, epk, world, dev)
        R = sum(recv_counts)

        flat_tok = torch.arange(P, device=dev) // K         # 每对所属 token
        send_x = x.reshape(T, H)[flat_tok[ord_]]            # [P, H] 派发序
        send_e = flat_e[ord_]                               # [P]

        recv_x = torch.empty(R, H, dtype=x.dtype, device=dev)
        if world > 1:
            dist.all_to_all_single(recv_x, send_x.contiguous(),
                                   output_split_sizes=recv_counts,
                                   input_split_sizes=send_counts,
                                   group=_get_ste_pg_hccl())
        else:
            recv_x.copy_(send_x)
        recv_e = torch.empty(R, dtype=torch.int64, device=dev)
        if world > 1:
            dist.all_to_all_single(recv_e, send_e.contiguous(),
                                   output_split_sizes=recv_counts,
                                   input_split_sizes=send_counts,
                                   group=_get_ste_pg_hccl())
        else:
            recv_e.copy_(send_e)
        e_local = recv_e % epk
        rows = torch.arange(R, device=dev)

        # 冗余计算：本 rank 全部 epk 个专家 × 全部收到的对，再 gather 选中项
        gu_all = (recv_x @ w_gu_eff.to(torch.bfloat16).reshape(epk * 2 * i_dim, H).t())
        gu_all = gu_all.view(R, epk, 2 * i_dim)
        gu_sel = gu_all[rows, e_local]                      # [R, 2i]
        gate, up = gu_sel.chunk(2, dim=-1)
        sig = torch.sigmoid(gate)
        h = sig * gate * up                                 # SiLU(gate)*up

        y_all = (h @ w_dn_eff.to(torch.bfloat16).reshape(epk * H, i_dim).t())
        y_all = y_all.view(R, epk, H)
        y_sel = y_all[rows, e_local]                        # [R, H]

        recv_y = torch.empty(P, H, dtype=x.dtype, device=dev)
        if world > 1:
            dist.all_to_all_single(recv_y, y_sel.contiguous(),
                                   output_split_sizes=send_counts,
                                   input_split_sizes=recv_counts,
                                   group=_get_ste_pg_hccl())
        else:
            recv_y.copy_(y_sel)
        out_flat = torch.empty(P, H, dtype=x.dtype, device=dev)
        out_flat[ord_] = recv_y                             # 派发序 → 自然序
        out = (out_flat.view(T, K, H) * topk_w.to(torch.bfloat16).unsqueeze(-1)).sum(dim=1)

        ctx.save_for_backward(ord_, flat_tok, e_local, recv_x, gate, up, h,
                              w_gu_eff, w_dn_eff, topk_w)
        ctx.send_counts, ctx.recv_counts = send_counts, recv_counts
        ctx.epk, ctx.world = epk, world
        ctx.T, ctx.K, ctx.H, ctx.i_dim = T, K, H, i_dim
        return out.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        import torch.distributed as dist
        ord_, flat_tok, e_local, recv_x, gate, up, h, w_gu_eff, w_dn_eff, topk_w = \
            ctx.saved_tensors
        epk, world = ctx.epk, ctx.world
        T, K, H, i_dim = ctx.T, ctx.K, ctx.H, ctx.i_dim
        P, R = T * K, recv_x.shape[0]
        dev = grad_out.device
        send_counts, recv_counts = ctx.send_counts, ctx.recv_counts
        go = grad_out.to(torch.bfloat16)

        # out = Σ_k w·y → ∂y[t,k] = grad_out[t]·w[t,k]；forward scatter ↔ backward gather
        grad_y_flat = (go.unsqueeze(1) * topk_w.to(torch.bfloat16).unsqueeze(-1)).reshape(P, H)
        grad_recv_y = grad_y_flat[ord_]                     # [P, H] 派发序
        grad_y_sel = torch.empty(R, H, dtype=torch.bfloat16, device=dev)
        if world > 1:
            dist.all_to_all_single(grad_y_sel, grad_recv_y.contiguous(),
                                   output_split_sizes=recv_counts,
                                   input_split_sizes=send_counts,
                                   group=_get_ste_pg_hccl())
        else:
            grad_y_sel.copy_(grad_recv_y)

        rows = torch.arange(R, device=dev)
        grad_y_all = torch.zeros(R, epk, H, dtype=torch.bfloat16, device=dev)
        grad_y_all[rows, e_local] = grad_y_sel
        w_dn_b = w_dn_eff.to(torch.bfloat16).reshape(epk * H, i_dim)
        grad_w_dn_eff = (grad_y_all.reshape(R, epk * H).t() @ h).view(epk, H, i_dim)
        grad_h = grad_y_all.reshape(R, epk * H) @ w_dn_b    # [R, i]

        sig = torch.sigmoid(gate)
        grad_gate = grad_h * up * (sig + gate * sig * (1 - sig))
        grad_up = grad_h * (sig * gate)
        grad_gu_sel = torch.cat([grad_gate, grad_up], dim=-1)   # [R, 2i]
        grad_gu_all = torch.zeros(R, epk, 2 * i_dim, dtype=torch.bfloat16, device=dev)
        grad_gu_all[rows, e_local] = grad_gu_sel
        w_gu_b = w_gu_eff.to(torch.bfloat16).reshape(epk * 2 * i_dim, H)
        grad_w_gu_eff = (grad_gu_all.reshape(R, epk * 2 * i_dim).t() @ recv_x).view(epk, 2 * i_dim, H)
        grad_recv_x = grad_gu_all.reshape(R, epk * 2 * i_dim) @ w_gu_b   # [R, H]

        grad_pairs = torch.empty(P, H, dtype=torch.bfloat16, device=dev)
        if world > 1:
            dist.all_to_all_single(grad_pairs, grad_recv_x.contiguous(),
                                   output_split_sizes=send_counts,
                                   input_split_sizes=recv_counts,
                                   group=_get_ste_pg_hccl())
        else:
            grad_pairs.copy_(grad_recv_x)
        grad_hidden = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        grad_hidden.index_add_(0, flat_tok[ord_], grad_pairs)

        grad_topk_w = None
        if ctx.needs_input_grad[2]:
            # ∂out[t]/∂w[t,k] = grad_out[t]·y[t,k]；y[t,k] 需从 out_flat 反解（未存），
            # QAD 路由权重恒冻结（requires_grad=False），此分支仅为完备性占位。
            grad_topk_w = torch.zeros_like(topk_w)
        return grad_hidden.to(grad_out.dtype), None, grad_topk_w, \
            grad_w_gu_eff.to(w_gu_eff.dtype), grad_w_dn_eff.to(w_dn_eff.dtype), None, None


class STEQwen3Experts(nn.Module):
    """包装 Qwen3MoeExperts：STE fake-quant 权重 + per-expert LoRA delta。

    用法：`wrapped = STEQwen3Experts(original_experts, rank=64)` 后把
    `layer.mlp.experts` 替换为 wrapped（保留原模块引用用于权重冻结）。
    """

    def __init__(self, experts: nn.Module, rank: int = 64,
                 dtype: torch.dtype = torch.float32, vectorized: bool = True,
                 use_checkpoint: bool = False, zero3_gather: bool = False,
                 deploy_scales: dict | None = None):
        super().__init__()
        self.experts = experts                      # 原 Qwen3MoeExperts（权重冻结）
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = experts.act_fn
        self.vectorized = vectorized
        self.use_checkpoint = use_checkpoint
        self.zero3_gather = zero3_gather
        # QAD 对齐（0908 §4.8-F / §5.13）：{"gate_up":[E,2I,1], "down":[E,H,1]} 为
        # P1 部署产物的专家 SSZ scale；None 则退回现场 minmax RTN（旧行为）。
        self.deploy_scales = deploy_scales
        # 任务 2（§5.14-B1 scale-only 对照臂）：QAD_TRAIN_SCALES=1 时把部署 scale
        # 注册为 Parameter（初始 = 产物 SSZ scale；wq = s·v 图内可导，LSQ 式梯度）。
        # 注意：在 zero.Init 之外创建 → 复制态（非分区），依赖 DS 梯度 all-reduce。
        self.scale_params = nn.ParameterDict()
        if deploy_scales and os.environ.get("QAD_TRAIN_SCALES") == "1":
            for proj, v in deploy_scales.items():
                self.scale_params[proj] = nn.Parameter(
                    v.detach().clone().float(), requires_grad=True)
        self.lora = ExpertLoRA(num_experts=self.num_experts,
                               hidden_size=self.hidden_dim,
                               moe_intermediate_size=self.intermediate_dim,
                               rank=rank, dtype=dtype)
        # S2 迁移：自管专家权重快照机制已移除（_effective_all 改读 LIVE engine
        # ds_tensor，调研文档 §10.5 已证 forward 期数值正确）。保留空 dict 与
        # no-op snapshot_ds_tensors() 仅作历史脚本（ds_train_t7/tiny_glm）向后
        # 兼容——它们的 .get(...) 诊断访问不会 AttributeError，但永远拿到 None。
        self._ds_snapshot: dict = {}
        self._ds_cpu: dict = {}
        self._wq_cache: dict | None = None
        self._wq_cache_enabled = False
        self._pf_cache: dict | None = None
        self._pf_enabled = os.environ.get("QAD_PREFETCH") == "1"
        self._pf_pin_warned = False
        # 复合顺序（QAD 文档 §8）：post（默认，historic）= fakequant(W)+δ；
        # deploy = fakequant(W+δ)（与部署 merge→requant 的复合一致，δ 经 STE 直通保梯度）
        self._compose = os.environ.get("QAD_COMPOSE", "post").lower()

    def _effective_all(self, proj: str) -> torch.Tensor:
        """全专家有效权重 = fakequant(W) + LoRA delta（[E, out, in] bf16）。

        S2 路径（§6e-5 收敛）：直接读 LIVE engine-managed ds_tensor
        （_zero3_gather_full 或非 ZeRO-3 的 w.detach），不再使用自管快照
        （_ds_cpu/_ds_snapshot 已废弃为空；调研文档 §10.5 已证 forward 期
        ds_tensor 数值正确，maxdiff=0 bit-exact vs safetensors）。

        内存约束（本次迁移核心）：逐专家 slab 流式计算。预分配 bf16 输出缓冲
        一次；每 slab 的 fp32 中间量（w/scale、round、clamp）只占单 slab 大小，
        全 [E,2I,H] fp32 中间量不再同时存在（旧实现 ~51.6 GB fp32 瞬态消失）。
        非训练 scale 路径直接复用 gather 缓冲为输出（in-place），峰值 ≈ 输出
        缓冲 + 单 slab；训练 scale 路径额外保留 gather 缓冲（fp32 wq 与 bf16 d
        分别相加后转 bf16，逐 bit 匹配旧 (wq_fp32+d).to(bf16)）。

        数值等价：fake-quant 与 einsum 均逐元素/逐专家独立，分片与全量计算
        逐 bit 一致（maxdiff=0）。可导性不变：wq 在 no_grad 内构造（常数，
        scale 梯度按旧实现不回传——与历史行为一致），d 在 no_grad 外构造，
        梯度经 d → LoRA b,a。
        """
        if proj == "gate_up":
            w = self.experts.gate_up_proj
            b, a = self.lora.params["Bgu"], self.lora.params["Agu"]
        else:
            w = self.experts.down_proj
            b, a = self.lora.params["Bdn"], self.lora.params["Adn"]

        # 1. 权威权重（LIVE engine-managed）
        if self.zero3_gather and getattr(w, "ds_tensor", None) is not None:
            # ZeRO-3 offload_param.device=cpu 时 ds_tensor 在 CPU：gloo all_gather
            # 重组后必须显式搬 NPU（下游 matmul/量化全在 NPU；probe S2 同款）
            wbuf = _zero3_gather_full(w).to(w.device)
        else:
            wbuf = w.detach().to(torch.bfloat16)
            if wbuf.data_ptr() == w.data_ptr():        # w 已是 bf16：to() 返回自身 → 克隆，避免 in-place 污染冻结权重
                wbuf = wbuf.clone()

        # 2. scale [E, out, 1]（小张量，一次性取定；逐 slab 切片）
        scale_trainable = proj in self.scale_params
        if scale_trainable:
            scale_full = self.scale_params[proj]
        elif self.deploy_scales is not None and self.deploy_scales.get(proj) is not None:
            scale_full = self.deploy_scales[proj].to(wbuf.device)
        else:
            scale_full = compute_weight_scale(wbuf, bits=4).to(wbuf.device)

        # 3. 输出缓冲：非训练 scale 复用 wbuf（in-place 省一份满张量）；训练 scale 独立（保留 fp32 加法精度）
        out = torch.empty_like(wbuf) if scale_trainable else wbuf

        # 4. 逐专家 slab 流式：wq（no_grad 常数）+ d（图内，梯度→b,a）→ 写入 out
        E = self.num_experts
        SLAB = _SLAB
        for e0 in range(0, E, SLAB):
            e1 = min(e0 + SLAB, E)
            with torch.no_grad():
                w_slab = wbuf[e0:e1]
                s_slab = scale_full[e0:e1]
                if scale_trainable:
                    v = torch.clamp(torch.round(w_slab / s_slab), -8, 7)
                    wq = s_slab * v                    # fp32（匹配旧 wq=scale*v）
                else:
                    wq = fake_quant_weight(w_slab, bits=4, scale=s_slab)
            d_slab = torch.einsum("eor,eri->eoi", b[e0:e1], a[e0:e1]) * self.lora.scaling
            if self._compose == "deploy":
                w32 = w_slab.float().clone()  # clone 断开 no_grad 版本链（原地 add_ 合法，0925 修复）
                w32.add_(d_slab)              # 原地加 δ（省一个 fp32 满张量临时）
                # 0925 修复 3b：切片赋值自带 fused cast（copy_ 语义），省 .to(bf16) 的
                # 满 slab 临时。out(wbuf) 为 bf16，fp32 结果直接 copy_ 舍入——bit 同旧路径。
                # 【0928 池化 Post-mortem】w32/v 池化已尝试并回退：add_ 原地操作把相邻
                # 调用/相邻 step 的 autograd 图经 pool grad_fn 链成一体，优化器对 LoRA
                # 参数的 in-place 更新毒化链上的 einsum saved-views → 版本错误
                # （test_compose_deploy 拦截）。clone 是图隔离的承重墙，不可池化。
                out[e0:e1] = fake_quant_weight(w32, bits=4, scale=s_slab)
            else:
                out[e0:e1] = (wq + d_slab.to(torch.bfloat16)).to(torch.bfloat16)
        return out

    def snapshot_ds_tensors(self):
        """已废弃（S2 迁移）。保留为 no-op 供历史脚本（ds_train_t7/tiny_glm）调用不报错。"""
        return

    # ---- 专家并行前向（Phase-2 2a'，QAD_EP=1）----
    # 动机：dense 路径每层每 forward 都要 all_gather 全部 256 专家权重
    # （gate_up 12.9GB + down 6.4GB），world=32 跨 4 节点 HCCL 带宽下
    # ~10s/次 × 300 次/step ≈ 50 min/step。EP 路径：每个 rank 只从自己的
    # ds_tensor 分片物化本 rank 的 8 个专家有效权重（无权重 all_gather），
    # 算完本 rank 专家的部分输出后 all_gather 输出（[T,E,2i]≈1GB /
    # [T,E,H]≈1.6GB，比权重 gather 小 7-13×）。fake-quant、LoRA einsum、
    # matmul/bmm 均逐专家独立 → 与 dense 路径数值等价（world=1 时逐 bit 一致）。

    def _ep_setup(self):
        """返回 (rank, world, epk)（EP 可用时）或 None（回退 dense 路径）。

        EP 可用条件：QAD_EP=1、zero3_gather、ds_tensor 存在、ZeRO-3 分片按
        专家对齐（partition = epk × expert_size，256/32=8 整除）、deploy_scales
        齐备（minmax 回退需全量权重，EP 不支持）、无 trainable scale。
        """
        if getattr(self, "_ep_checked", False):
            return self._ep_info
        self._ep_checked = True
        self._ep_info = None
        if os.environ.get("QAD_EP") != "1" or not self.zero3_gather:
            return None
        if self.scale_params:
            return None
        w = self.experts.gate_up_proj
        dt = getattr(w, "ds_tensor", None)
        dn = self.experts.down_proj
        if dt is None or getattr(dn, "ds_tensor", None) is None:
            return None
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world, rank = dist.get_world_size(), dist.get_rank()
        else:
            world, rank = 1, 0
        E = self.num_experts
        if E % world != 0:
            return None
        epk = E // world
        gu_exp = w.ds_numel // E
        dn_exp = dn.ds_numel // E
        if (w.ds_numel % E or dn.ds_numel % E
                or dt.numel() != epk * gu_exp
                or dn.ds_tensor.numel() != epk * dn_exp):
            return None
        if (self.deploy_scales is None
                or self.deploy_scales.get("gate_up") is None
                or self.deploy_scales.get("down") is None):
            return None
        self._wq_cache_enabled = os.environ.get("QAD_WQ_CACHE", "0") == "1"
        # 复合顺序（§QAD 文档 §8）：post（默认，historic）= fakequant(W)+δ；
        # deploy = fakequant(W+δ)（与部署 merge→requant 的复合一致，δ 经 STE 直通保梯度）
        self._compose = os.environ.get("QAD_COMPOSE", "post").lower()
        self._ep_info = (rank, world, epk)
        return self._ep_info

    def _effective_shard(self, proj: str, w_shard: torch.Tensor,
                         e0: int, epk: int) -> torch.Tensor:
        """本 rank 专家 [e0, e0+epk) 的有效权重 [epk, out, in] bf16。

        与 _effective_all 的数学逐 bit 一致（fake-quant 与 LoRA einsum 均逐
        专家独立，shard 切片不改变单专家计算）；wq 在 no_grad 内构造（scale
        梯度不回传，与历史一致），d 在 no_grad 外构造（梯度经 d → b,a）。
        """
        if proj == "gate_up":
            b, a = self.lora.params["Bgu"], self.lora.params["Agu"]
        else:
            b, a = self.lora.params["Bdn"], self.lora.params["Adn"]
        scale_trainable = proj in self.scale_params
        if scale_trainable:
            scale_full = self.scale_params[proj]
        else:
            scale_full = self.deploy_scales[proj].to(w_shard.device)
        out = torch.empty_like(w_shard)
        SLAB = _SLAB
        for j0 in range(0, epk, SLAB):
            j1 = min(j0 + SLAB, epk)
            with torch.no_grad():
                w_slab = w_shard[j0:j1]
                s_slab = scale_full[e0 + j0:e0 + j1]
                if self._compose != "deploy":
                    if scale_trainable:
                        v = torch.clamp(torch.round(w_slab / s_slab), -8, 7)
                        wq = s_slab * v
                    else:
                        wq = fake_quant_weight(w_slab, bits=4, scale=s_slab)
            d_slab = torch.einsum("eor,eri->eoi",
                                  b[e0 + j0:e0 + j1], a[e0 + j0:e0 + j1]) * self.lora.scaling
            if self._compose == "deploy":
                w32 = w_slab.float().clone()  # clone 断开 no_grad 版本链（原地 add_ 合法，0925 修复）
                w32.add_(d_slab)              # 原地加 δ
                # 0925 修复 3b：切片赋值自带 fused cast（copy_ 语义），省 .to(bf16) 的
                # 满 slab 临时。out 与 w_shard 同 dtype（fp32），下游 matmul 处统一
                # .to(bf16)——最终 matmul 输入 bit 不变（fp32→bf16 舍入点后移一次）。
                # 【0928 池化 Post-mortem】w32/v 池化已尝试并回退（同 _effective_all 注）。
                out[j0:j1] = fake_quant_weight(w32, bits=4, scale=s_slab)
            else:
                out[j0:j1] = (wq + d_slab.to(torch.bfloat16)).to(torch.bfloat16)
        return out

    def _effective_shard_prefetch(self, proj: str, w_shard: torch.Tensor,
                                  e0: int, epk: int, device) -> torch.Tensor:
        """码本（int4 原码）常驻 pinned host + 每层 non_blocking async H2D（QAD_PREFETCH=1）。

        动机（profiler 实测）：OFFLOAD_PARAM=1 下逐层逐步 bf16 阻塞拷贝 ~47GB/步，host
        阻塞占 wall 45.7%（aten::copy_ 均 2.7ms；无 sync 时 41ms＝队列满停走）。本路径：
        ① 码本一次性 pack 到 pinned host（~11.3GB/rank，host RSS 57+12=69 < 80 看门狗）；
        ② 每层 non_blocking=True async H2D（拷贝量 47→11.3GB/步、host 不再被阻塞）；
        ③ on-device unpack wq=v*s。与 QAD_WQ_CACHE 的关键差异：**不占 HBM**（驻卡 11.8GB
        即当初 32 卡 OOM/thrash 根因）。数值与 _effective_shard 逐 bit 等价（同式同 dtype）。

        0923 泄漏修复（staging 复用）：原实现每次调用新分配 int8 H2D 目标 + bf16 cast +
        mul 结果（~1.5GB/次 × ~150 次/步 ≈ 225GB allocator 流量）→ expandable_segments
        碎片化 + cache flush 爬升（phase2b 2000 步 1 次 → smoke 每步 2-3 次）。改为
        (proj,device) 级共享 bf16 暂存：copy_(cast)+mul 全 in-place（零新分配）；返回值
        仍为 fresh 张量（autograd 所需，与 _effective_shard 同）。别名安全：stg 仅在
        本调用内被消费（下游读 fresh 返回值），同流顺序化复用；step N 的 bwd 完成
        后才进 step N+1 的 fwd，无跨期活引用。
        """
        cache = self._pf_cache
        if cache is None:
            cache = self._pf_cache = {}
        ent = cache.get(proj)
        if ent is None:
            w_cpu = w_shard if w_shard.device.type == "cpu" else w_shard.to("cpu")
            s_cpu = self.deploy_scales[proj][e0:e0 + epk].to("cpu", torch.bfloat16)
            with torch.no_grad():
                v = torch.clamp(torch.round(w_cpu / s_cpu), -8, 7).to(torch.int8)
            try:
                v = v.pin_memory()
            except Exception as exc:  # noqa: BLE001 pin 不可用则退化（语义弱化、数值不变）
                if not self._pf_pin_warned:
                    self._pf_pin_warned = True
                    print(f"[qad] pin_memory unavailable ({exc}); pageable fallback",
                          flush=True)
            ent = (v, s_cpu.to(device))
            cache[proj] = ent
        codes, s = ent
        skey = (proj, str(device))
        stg = _PF_STAGING.get(skey)
        if stg is None:
            stg = torch.empty(codes.shape, dtype=torch.bfloat16, device=device)
            _PF_STAGING[skey] = stg
        stg.copy_(codes, non_blocking=True)   # H2D + int8→bf16 cast（单 kernel，零新分配）
        torch.mul(stg, s, out=stg)             # in-place → stg = wq（零新分配）
        if proj == "gate_up":
            b, a = self.lora.params["Bgu"], self.lora.params["Agu"]
        else:
            b, a = self.lora.params["Bdn"], self.lora.params["Adn"]
        d = torch.einsum("eor,eri->eoi", b[e0:e0 + epk], a[e0:e0 + epk]) \
            * self.lora.scaling
        return (stg + d.to(torch.bfloat16)).to(torch.bfloat16)

    def _effective_shard_cached(self, proj: str, w_shard: torch.Tensor,
                                e0: int, epk: int, target_device) -> torch.Tensor:
        """int4 码本驻卡版 _effective_shard（QAD_WQ_CACHE=1 显式开启；默认关）。

        逐 bit 等价性：v = clamp(round(w / s_bf16)) 与 _STEQuant.forward 同式同
        dtype（s 先 to(w.dtype)=bf16，除法在 bf16 内做），wq = v*s 同式；W 冻结
        → 码本跨步恒定，仅 LoRA delta 逐步变化。首访一次性把本 rank 专家的
        bf16 去量化权重转 int4 码（offset-binary uint8 nibble 打包，v+8 ∈ [0,15]）
        + bf16 scale 驻卡（~11.8GB/rank），每步 H2D 拷贝归零——单机小模型上可
        消除 OFFLOAD_PARAM=1 的阻塞拷贝（profiler 实测 aten::copy_ 占 wall 70%）。
        ★ 32 卡实测（09-20）：码本驻卡使 reserved 逼近 61GB 墙（int8 23.5GB 直接
        OOM；int4 11.8GB 无 sync 时分配器 thrash ~5min/步；+QAD_DIAG sync 后
        a2a 吸收 rank 偏移 1.7→48-90s，净劣化 2-3×）→ 32 卡默认关闭，保留作
        小规模/单卡加速选项。
        """
        if self._wq_cache is None:
            self._wq_cache = {}
        ent = self._wq_cache.get(proj)
        if ent is None:
            w_dev = w_shard if w_shard.device == target_device \
                else w_shard.to(target_device)
            s = self.deploy_scales[proj][e0:e0 + epk].to(
                device=target_device, dtype=torch.bfloat16)
            with torch.no_grad():
                assert w_dev.shape[-1] % 2 == 0, \
                    f"nibble 打包要求最后一维为偶数，got {w_dev.shape}"
                v = torch.clamp(torch.round(w_dev / s), -8, 7).to(torch.int8) + 8
                lo = v[..., 0::2].to(torch.uint8)
                hi = v[..., 1::2].to(torch.uint8)
                codes = (lo << 4) | hi
            ent = (codes, s)
            self._wq_cache[proj] = ent
        codes, s = ent
        lo = (codes >> 4).to(torch.int8) - 8
        hi = (codes & 0xF).to(torch.int8) - 8
        v = torch.stack((lo.to(torch.bfloat16), hi.to(torch.bfloat16)), dim=-1)
        v = v.flatten(-2)
        wq = v * s
        if proj == "gate_up":
            b, a = self.lora.params["Bgu"], self.lora.params["Agu"]
        else:
            b, a = self.lora.params["Bdn"], self.lora.params["Adn"]
        d = torch.einsum("eor,eri->eoi", b[e0:e0 + epk], a[e0:e0 + epk]) \
            * self.lora.scaling
        return (wq + d.to(torch.bfloat16)).to(torch.bfloat16)


    def _forward_vec_ep(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        """专家并行前向（sparse dispatch 单片 Function，数值等价 dense）。

        权重侧：本 rank 只物化自己 8 个专家的有效权重（_effective_shard，无
        权重 all_gather）；通信侧：隐状态/输出按 (token,slot) 对稀疏派发
        （~50MB/跳 vs dense 权重 gather 12.9GB）。梯度经 _EpExpertBlock 的
        反向 all_to_all 回流到 LoRA。
        """
        rank, world, epk = self._ep_info
        e0 = rank * epk
        i_dim, h_dim = self.intermediate_dim, self.hidden_dim
        x = hidden_states.to(torch.bfloat16)

        w_gu = self.experts.gate_up_proj
        w_shard = w_gu.ds_tensor.view(epk, 2 * i_dim, h_dim)
        w_dn = self.experts.down_proj
        w_dn_shard = w_dn.ds_tensor.view(epk, h_dim, i_dim)
        if self._pf_enabled and self._compose != "deploy":
            w_gu_eff = self._effective_shard_prefetch("gate_up", w_shard, e0, epk, x.device)
            w_dn_eff = self._effective_shard_prefetch("down", w_dn_shard, e0, epk, x.device)
        elif self._wq_cache_enabled and self._compose != "deploy":
            w_gu_eff = self._effective_shard_cached("gate_up", w_shard, e0, epk, x.device)
            w_dn_eff = self._effective_shard_cached("down", w_dn_shard, e0, epk, x.device)
        else:
            if w_shard.device.type == "cpu":
                w_shard = w_shard.to(x.device)
            if w_dn_shard.device.type == "cpu":
                w_dn_shard = w_dn_shard.to(x.device)
            w_gu_eff = self._effective_shard("gate_up", w_shard, e0, epk)
            w_dn_eff = self._effective_shard("down", w_dn_shard, e0, epk)

        return _EpExpertBlock.apply(hidden_states, top_k_index, top_k_weights,
                                    w_gu_eff, w_dn_eff, self.act_fn, epk)

    def forward(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor:
        if self.vectorized:
            fwd = self._forward_vec_ep if self._ep_setup() is not None else self._forward_vec
            if self.use_checkpoint:
                # 反向重算 STE 有效权重与中间量，避免 48 层 × fp32 权重被 autograd
                # 保存导致 OOM（FSDP 场景必须开）
                return torch.utils.checkpoint.checkpoint(
                    fwd, hidden_states, top_k_index, top_k_weights,
                    use_reentrant=False)
            return fwd(hidden_states, top_k_index, top_k_weights)
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


def wrap_all_experts_glm(model, rank: int = 64, vectorized: bool = True,
                         dtype: torch.dtype = torch.float32,
                         use_checkpoint: bool = False,
                         zero3_gather: bool = False,
                         scale_provider=None) -> list[STEQwen3Experts]:
    """给 GLM-5.2 (GlmMoeDsa) 的 MoE 专家层套 STE+LoRA 包装（C1 正式化）。

    GLM 与 Qwen3MoE 的差异（见 workspace/glm52-qad/ds_train_tiny_glm.py）：
      - MoE 层为 GlmMoeDsaNaiveMoe：属性与 Qwen3MoeExperts 同构
        （num_experts/hidden_dim/intermediate_dim/act_fn/gate_up_proj/down_proj），
        故直接复用 STEQwen3Experts（tiny 预演已证同构可复用）；
      - dense MLP 层（GlmMoeDsaMLP，前 first_k_dense_replace 层）无 .experts → 跳过。

    scale_provider(layer_idx) -> {"gate_up":[E,2I,1],"down":[E,H,1]} 时，用 P1 部署产物的
    专家 SSZ scale 做 fake-quant 对齐（QAD Phase-1）；None 则现场 minmax RTN。

    返回仅含实际包装的 MoE 层列表；冻结原专家权重（W 不训练），LoRA 参数可训练。
    """
    wrapped = []
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "experts"):
            continue                           # dense MLP 层（GlmMoeDsaMLP）跳过
        exp = mlp.experts
        for p in exp.parameters():
            p.requires_grad = False
        deploy_scales = scale_provider(layer_idx) if scale_provider is not None else None
        w = STEQwen3Experts(exp, rank=rank, vectorized=vectorized, dtype=dtype,
                            use_checkpoint=use_checkpoint, zero3_gather=zero3_gather,
                            deploy_scales=deploy_scales)
        w.to(exp.gate_up_proj.device)
        mlp.experts = w
        wrapped.append(w)
    return wrapped
