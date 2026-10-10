"""M4 DeepSpeed ZeRO-3 冒烟训练（8 NPU）：替代 FSDP 方案（Ascend 适配）。

背景：torch_npu 2.10 标准 FSDP backward 阶段 HcclAllreduce OOM（gather 权重被
autograd graph 长期引用 + 通信缓冲过大）。改用 DeepSpeed ZeRO-3：

并行设计：
- 模型经 transformers zero.Init 路径加载：所有参数在构造时即参数级分片到 8 卡
  （engine 源码确认 is_zero_init_model 时跳过 module.to(device)，避免 60GB
  bf16 模型整体 .to(npu) 的瞬时 OOM）
- 专家大权重（gate_up 402M / down 201M per proj）分片驻留，STE 包装器绕过
  experts.forward → ZeRO-3 自动 gather hook 不触发 → _effective_all 显式
  重组权重。方案 A（2026-09-07）：engine.initialize 后、首次 forward 前
  对 ds_tensor 做私有快照并在 forward 中读快照——checksum 探针实证 forward
  期间 ds_tensor 被引擎重指向（调研文档 §10.3），现场重组不可用
- LoRA/attention/router 小张量（rank16 时 ≤8.4M 元素）经
  stage3_param_persistence_threshold=25M 保持每卡全量副本：正常 autograd、
  无需 gather（与 FSDP 方案"attention 每卡复制"一致）
- embed/lm_head（311M 元素）分片；数据并行：每 rank 独立 micro-batch
- 层级 activation checkpointing：backward 重算层前向，逐层 gather/释放

用法（容器内，torchrun 8 进程）：
    DS_ACCELERATOR=npu python -m torch.distributed.run --nproc_per_node=8 \
        ds_train_t7.py [--steps N] [--micro-tokens L] [--rank R] [--lr L]
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

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, f"{SHARED_HOST}/ylt/npuslim/src")
sys.path.insert(0, f"{SHARED_HOST}/ylt/qwen3-qad/precompute")

from npuslim.qad.teacher_cache import TeacherCache  # noqa: E402
from npuslim.qad.kl_loss import kl_distill_loss    # noqa: E402
from npuslim.qad.ste_experts import STEQwen3Experts, wrap_all_experts  # noqa: E402
from precompute_logits import encode_one            # noqa: E402

MODEL_DIR = f"{SHARED_HOST}/lichc/models/Qwen3-30B-A3B-Instruct-2507"
CACHE = Path(f"{SHARED_HOST}/ylt/qwen3-qad/precompute/logits_cache_topk100")
TRAIN = Path(f"{SHARED_HOST}/ylt/glm52-qad/qad_data/qad_train_v1.jsonl")
CORPUS = Path(f"{SHARED_HOST}/ylt/glm52-qad/glm52-qad-corpus/"
              "processed/corpus_v1_2026-08-27/clean")
VOCAB = 151_936
VCHUNK = 65536


def build_ds_config(world: int, lr: float) -> dict:
    # 注意：不显式设 train_batch_size —— zero.Init 阶段 DeepSpeedConfig 的
    # world_size=1，engine 阶段=world，两阶段各自按 micro×ga×world 推导才自洽
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            # 禁用默认 leaf module（Qwen3MoeSparseMoeBlock 等）：该机制面向 DeepSpeed
            # MoE 专家并行；我们无 EP 且专家权重已冻结，leaf hook 会对冻结的 402M
            # 专家权重 zeros_like 并重复入梯度桶（IPG ds_id 断言失败）
            "leaf_module": {"classes": []},
            # LoRA(rank16 ≤4.2M)/attention/router 张量保持全量副本；专家大权重分片
            "stage3_param_persistence_threshold": 25_000_000,
            # 其余 gather 相关（prefetch/max_live/max_reuse）用 DeepSpeed 默认值：
            # 自定义 prefetch=0/max_reuse=0 下 hook gather 桶写入错层数据（见
            # --checksum 探针 2026-09-07），先回归默认验证
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        },
        "optimizer": {"type": "AdamW",
                      "params": {"lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.0}},
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro-tokens", type=int, default=512)
    ap.add_argument("--min-enc-len", type=int, default=256,
                    help="样本编码长度下限：步时间被固定开销（权重重组/checkpoint/"
                         "优化器）主导，超短样本每 token 成本差 ~20×（2026-09-07 "
                         "100 步冒烟实测：nemotron-pt 短行 L_use≈25 → 全局仅 "
                         "28 tok/s）。全量 epoch 组样建议同阈值")
    ap.add_argument("--rank", type=int, default=16)   # LoRA rank
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--skip-fake-quant", action="store_true",
                    help="调试探针：跳过 fake-quant（LoRA B=0 → 学生=教师），"
                         "loss 应≈0；用于验证 ZeRO-3 gather 是否为真实权重")
    ap.add_argument("--checksum", action="store_true",
                    help="调试探针：gather layer0 gate_up 与 safetensors 原值对拍后退出")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 步保存 LoRA checkpoint（0=关闭）；全量训练必开——"
                         "40h 级运行无断点，遇 NPU 瞬时故障/他人抢卡全部归零")
    ap.add_argument("--save-dir", default="ckpt_m4")
    ap.add_argument("--resume", default="",
                    help="断点续训：ckpt 文件路径；注意须与原 run 相同 --steps/"
                         "--micro-tokens（决定 stride 采样与生成器序，保证数据序一致）")
    args = ap.parse_args()

    if args.skip_fake_quant:
        import npuslim.qad.ste_experts as _ste_exp
        _ste_exp.fake_quant_weight = lambda w, bits=4, scale=None: w
        if int(os.environ.get("RANK", 0)) == 0:
            print("  [probe] fake-quant 已禁用（学生=教师）", flush=True)

    # timeout 放宽：预取耗时长且方差大（冒烟 stride 采样分散在 ~100 个 shard，
    # 2-6 min/rank），HCCL communicator 在首个集合通信才惰性建立，慢 rank 未到
    # 会 EI0015 连接超时（2026-09-07 实测 ~4.5min 崩）——c10d 层超时同步放宽
    dist.init_process_group(backend="hccl",
                            timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.npu.set_device(local_rank)
    dev = f"npu:{local_rank}"
    if rank == 0:
        print(f"DS-ZeRO3 world={world} micro={args.micro_tokens} "
              f"total/step={world*args.micro_tokens}", flush=True)

    # ---- 数据管线：全局混洗 → rank interleaved 分片 → epoch 循环流式取样 ----
    # ① jsonl 按域分组存储（前 2 万行全是 agentic_coding），连续块分片会让各 rank
    #   域不均 → 先全局混洗（seed42，各 rank 同序）再 interleaved 条带分片；
    # ② 流式生成器逐样本取数（不再整池预取）：TeacherCache shard 惰性加载+LRU。
    #   2026-09-08 死锁复盘：整池预取在 min-enc-len 过滤后各 rank 产出数不齐
    #   （rank1/5 差 1 条），样本耗尽的 rank 提前 break、其余 rank 继续训练 →
    #   集合通信错序死锁；且全量 12000 步 > 每 rank 11408 样本，必须 epoch 循环；
    # ③ 训练序块内乱序（去相关）+ 块间保持 (shard,row) 聚簇（shard 顺序读保 LRU
    #   命中，乱序访问会反复重载 ~1.6GB shard）；
    # ④ 重复 id（1779 个，最大 27 条）由 fetch(expected_ids) 内容级消歧
    tc = TeacherCache(CACHE)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t0 = time.time()
    entries: list = []   # [(sample_key, id)] 全 rank 一致
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                entries.append((s["sample_key"], s["id"]))
    rng = random.Random(42)
    rng.shuffle(entries)
    mine = entries[rank::world]
    if rank == 0:
        print(f"  数据集 {len(entries)} 条；rank 分片 {len(mine)} 条", flush=True)
    # 冒烟选样：均匀 stride（在混洗序上，保证域/长度代表性）。不可取聚簇序前缀：
    # shard 按语料行序组织，前缀会整块落进同一 shard 的同长度区间——2026-09-07
    # 实测 rank0 前 100 条全是 E-general-nemotron-pt 的 19-20 token 短行 →
    # L_use≈25、全局吞吐失真 20×。n_need 只决定冒烟子集大小；生成器 epoch 循环
    # 永不耗尽（全量 stride 收敛为 1）
    n_need = max(args.steps, 4)
    pool = mine
    if len(pool) > n_need * 2:
        stride = max(1, len(pool) * 10 // (n_need * 13))
        pool = pool[::stride]
    order = {sid: (tc._by_id[sid][-1].shard, tc._by_id[sid][-1].row)
             for _, sid in pool if sid in tc._by_id}
    pool.sort(key=lambda e: order.get(e[1], ("", -1)))   # 仅 fetch 序聚簇
    line_cache: dict[str, list[str]] = {}

    BLOCK = 64   # 块内乱序窗口：块间聚簇保 LRU 局部性，块内乱序去训练序相关

    def fetch_stream():
        """epoch 循环流式取样：每 epoch 轮转起始点（off 步进 373，rotation 必
        遍历全池且保持 shard 聚簇局部性）。resume 依赖确定性：相同 args+seed 下
        序列完全可复现，续训第 k 步拿到与原 run 相同样本。"""
        n_pool = len(pool)
        brng = random.Random(1337)   # 块内乱序（确定性；勿与全局 rng 混用）
        epoch = 0
        while True:
            off = (epoch * 373) % n_pool
            for b0 in range(0, n_pool, BLOCK):
                block = [pool[(i + off) % n_pool] for i in range(b0, min(b0 + BLOCK, n_pool))]
                brng.shuffle(block)
                for key, sid in block:
                    fname, _, ln = key.rpartition(":")
                    if fname not in line_cache:
                        with open(CORPUS / fname, encoding="utf-8") as fh:
                            line_cache[fname] = fh.readlines()
                    row = json.loads(line_cache[fname][int(ln)].strip())
                    enc = encode_one(tok, row, 8192)
                    if not (enc is not None and not isinstance(enc, Exception) and len(enc) > 1):
                        continue
                    if len(enc) < args.min_enc_len:   # 短样本效率过滤，见 --min-enc-len help
                        continue
                    expected = np.asarray(enc[1:], dtype=np.int32)
                    if expected.shape[0] < 20:
                        continue
                    fs = tc.fetch(sid, expected_ids=expected)
                    # 学生输入 = 完整 token 序列（含 BOS）：teacher 行 i = P(enc[i+1] | enc[0..i])，
                    # 学生 logits 行 i 需看到 enc[0..i] 才与之对齐。actual_ids=enc[1:] 只是缓存
                    # 自校验流，直接喂入会 off-by-one（学生预测 enc[i+2] vs 教师 enc[i+1]），
                    # KL 掉到随机水平——见 probe_teacher_align.py A/B 对比（23.7 → -0.06）
                    L_enc = min(len(enc), args.micro_tokens)
                    L_use = min(fs.topk_ids.shape[0], L_enc - 1)
                    yield (np.asarray(enc, dtype=np.int64)[:L_enc],
                           fs.topk_ids[:L_use].copy(), fs.topk_lps[:L_use].copy())
            epoch += 1

    gen = fetch_stream()
    first = next(gen)   # 首样本预取（含首个 shard 冷加载）；逐 rank 差异由下方 barrier 吸收
    dt = time.time() - t0
    print(f"  [rank{rank}] 流式取样就绪 pool={len(pool)} 首样本 {dt:.1f}s", flush=True)
    # 对齐后再进模型加载：首个 HCCL 集合通信（zero.Init 分片加载内部）要求全部
    # rank 同时到场建域，慢 rank 预取未完成时快 rank 先到会建联超时（EI0015，
    # 2026-09-07 实测 rank1 预取 >4.6min 其余 112-202s，HCCL 建域 ~100s 即崩）。
    # 对齐本身必须用 gloo barrier：CPU 网络通信、不触发 HCCL 建域、可长时间等
    # 慢 rank（受 init timeout 30min 约束）；hccl barrier 只作兜底
    try:
        _gloo = dist.new_group(backend="gloo",
                               timeout=datetime.timedelta(minutes=30))
        dist.barrier(group=_gloo)
        if rank == 0:
            print("  gloo barrier 对齐完成", flush=True)
    except Exception as e:
        print(f"  [rank{rank}] gloo barrier 失败({e})，回退 hccl barrier", flush=True)
        dist.barrier()

    # ---- 模型：HfDeepSpeedConfig 激活 transformers zero3 路径（构造即分片） ----
    from transformers import AutoModelForCausalLM
    from transformers.integrations import HfDeepSpeedConfig
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    ds_config = build_ds_config(world, args.lr)
    dschf = HfDeepSpeedConfig(ds_config)   # 必须在 from_pretrained 前创建并保活
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    if rank == 0:
        print(f"  zero.Init 分片加载 {time.time()-t0:.0f}s", flush=True)

    # zero.Init 上下文已随 from_pretrained 结束 → STE 包装的 LoRA 是普通参数
    wrapped = wrap_all_experts(model, rank=args.rank, vectorized=True,
                               dtype=torch.bfloat16, use_checkpoint=False,
                               zero3_gather=True)
    for i, w in enumerate(wrapped):
        w._qad_idx = i               # 身份打桩：探针捕获时定位实际是哪一层
    for p in model.parameters():
        p.requires_grad = False
    for w in wrapped:
        for p in w.lora.params.values():
            p.requires_grad = True
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    # 层级 activation checkpointing：backward 重算层前向，逐层 gather/释放专家权重
    # 注意（2026-09-07 A3 探针实证，调研文档 §10.6）：闭包必须按层绑定 orig_fwd。
    # 原写法 orig_fwd 是闭包自由变量，循环结束被重绑到最后一层 → 全部 48 层
    # 实际执行 layer47 的 forward（events 全为 idx=47、skip-fq loss=13.61 的
    # 真正根因；此前观察到的"引擎 gather 错层"实为 layer47 正确数据 vs
    # layer0 参照的对比假象）
    def _make_ckpt_fwd(orig_fwd):
        def _ckpt_fwd(*a, **kw):
            return torch.utils.checkpoint.checkpoint(
                orig_fwd, *a, use_reentrant=False, **kw)
        return _ckpt_fwd

    for layer in model.model.layers:
        layer.forward = _make_ckpt_fwd(layer.forward)

    # ---- DeepSpeed 引擎（is_zero_init_model → 不做整体 .to(device)） ----
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)
    if rank == 0:
        from deepspeed.runtime.zero.partition_parameters import is_zero_param
        part = sum(1 for p in model.parameters()
                   if is_zero_param(p) and p.ds_status == ZeroParamStatus.NOT_AVAILABLE)
        persist = sum(1 for p in model.parameters()
                      if is_zero_param(p) and p.ds_status == ZeroParamStatus.AVAILABLE)
        normal = sum(1 for p in model.parameters() if not is_zero_param(p))
        print(f"  可训练 {n_train/1e6:.1f}M；参数状态: 分片{part} / 常驻{persist} / 普通 LoRA{normal}",
              flush=True)

    # ---- 方案 A（调研文档 §10.4）：私有快照 ----
    # checksum 探针（2026-09-07）实证：engine() forward 期间 ds_tensor 被引擎
    # 重指向（data_ptr 变化、内容错层，forward 后恢复）。initialize 后、首次
    # forward 前 clone 专家参数私有快照，_effective_all 在 forward 中改读快照。
    for w in wrapped:
        w.snapshot_ds_tensors()
    if rank == 0:
        n_snap = sum(len(w._ds_snapshot) for w in wrapped)
        print(f"  [fix-A] 专家参数 ds_tensor 私有快照 {n_snap}/{2 * len(wrapped)}",
              flush=True)

    # ---- Checkpoint（全量训练断点保护，40h 级运行无断点=全部归零风险）----
    # 只读原语手写保存：AVAILABLE 态直读 param.data（优化器刚写入的新值），
    # NOT_AVAILABLE 态读 ds_tensor 分片 all_gather 重组（与专家只读 gather 同路
    # 径，已验证安全）。engine.save_checkpoint 实测死锁（2026-09-08：64GB 文件
    # 写完但 rank 间集合通信错序，2 rank 空转 6 rank 阻塞）——zero.Init 混合参数
    # 状态下其内部状态机不同步，弃用。新鲜度自检：LoRA B 矩阵初始为 0，若保存
    # 值 zero_frac≈0.5 说明 ds_tensor 是陈旧初始值（训练被架空），必须报警
    lora_state = {}   # name -> 活参数对象（DS initialize 会用 PartitionedParameter
                      # 替换模块参数属性，pre-init 引用是孤儿对象，勿复用）
    for n, p in model.named_parameters():
        if p.requires_grad:
            lora_state[n] = p
    lora_names = sorted(lora_state)
    if args.save_every:
        os.makedirs(args.save_dir, exist_ok=True)

    def _local_shard_flat(p):
        """本 rank 权威分片（无跨 rank 状态依赖）：AVAILABLE 从全量 data 按 rank
        序切片（gathered buffer 即分片 rank 序 concat，与专家重组路径一致），
        否则读 ds_tensor。均为优化器更新后的值。"""
        dt = getattr(p, "ds_tensor", None)
        if dt is None:                       # 未被 DS 分片的普通参数（每 rank 全量同值）
            return p.detach().to(torch.bfloat16).contiguous().view(-1), True
        shard_n, r = dt.numel(), dist.get_rank()
        if p.ds_status == ZeroParamStatus.AVAILABLE and p.data.numel() == p.ds_numel:
            flat = p.detach().to(torch.bfloat16).contiguous().view(-1)
            lo = min(r * shard_n, max(p.ds_numel - shard_n, 0))
            return flat.narrow(0, lo, shard_n).contiguous(), True
        return dt.detach().contiguous(), False

    def save_ckpt(step_done: int, tag: str):
        """全 rank 集合调用；rank0 落盘（atomic tmp+rename）。
        关键（2026-09-08 死锁复盘）：集合通信模式必须与 per-rank 参数状态解耦——
        参数 gather/release 时序跨 rank 异步（rank1/5 滞后），此前按 status 分支
        导致 2 rank 进 all_gather、6 rank 进 barrier 的错序死锁。现在统一为：
        本地取分片（状态只影响读哪个源）+ 无条件 all_gather。"""
        sd, n_direct = {}, 0
        world = dist.get_world_size()
        for i, n in enumerate(lora_names):
            p = lora_state[n]
            shard, direct = _local_shard_flat(p)
            n_direct += int(direct)
            if getattr(p, "ds_tensor", None) is not None and world > 1:
                parts = [torch.empty_like(shard) for _ in range(world)]
                dist.all_gather(parts, shard)
                full = torch.cat(parts, dim=0)[:p.ds_numel]
            else:
                full = shard[:p.ds_numel]
            sd[n] = full.view(list(p.ds_shape)).float().cpu()
            del shard, full
            if i % 48 == 0:   # 死锁定位打点：逐 rank 的 gather 进度必须齐步
                print(f"  [rank{rank}] save {i}/{len(lora_names)}", flush=True)
        if rank == 0:
            zero_el = sum(int((t == 0).sum()) for t in sd.values())
            tot_el = sum(t.numel() for t in sd.values())
            stale = tot_el and zero_el > 0.2 * tot_el   # B 初始 0：新鲜≈0.5%，陈旧≈49%
            path = os.path.join(args.save_dir, f"step_{step_done:06d}.pt")
            tmp = path + ".tmp"
            torch.save({"step": step_done, "sd": sd, "args": vars(args)}, tmp)
            os.replace(tmp, path)            # 原子落盘，防半写文件被 supervisor 捡到
            print(f"  [ckpt] {tag} step={step_done} -> {path} ({len(sd)} tensors, "
                  f"direct_data={n_direct}/{len(sd)}, zero_frac={zero_el / max(tot_el, 1):.4f}"
                  f"{' WARN_STALE' if stale else ''})", flush=True)
        del sd
        dist.barrier()

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(ck["step"])
        for n in lora_names:
            p, dt = lora_state[n], getattr(lora_state[n], "ds_tensor", None)
            t = ck["sd"][n]
            if dt is None:
                p.data.copy_(t.to(p.dtype))
            else:
                shard_n = dt.numel()
                flat = t.view(-1).to(dt.dtype)
                lo = dist.get_rank() * shard_n
                dt.copy_(flat.narrow(0, lo, shard_n).view_as(dt))   # 预 forward 时
                # ds_tensor 是权威持久分片，首次 gather 从它载入
        del ck
        dist.barrier()
        if rank == 0:
            print(f"  [resume] 从 step={start_step} 续训（{args.resume}）",
                  flush=True)

    def mem_report(tag):
        a = torch.npu.memory_allocated(local_rank) / 2**30
        p = torch.npu.max_memory_allocated(local_rank) / 2**30
        print(f"  [mem:{tag} rank{rank}] alloc={a:.1f}GiB peak={p:.1f}GiB", flush=True)
        dist.barrier()

    # ---- 训练循环 ----
    if args.checksum:
        # ZeRO-3 权重正确性探针（checkpoint 用 per-expert 真名，融合序 gate→up）：
        # 1) 手动 GatheredParameters 与 safetensors 对拍
        # 2) 引擎 forward 中 hook 路径下 _effective_all 实际看到的权重对拍
        #    （rank0 视角看 e127：若 hook 未 gather，e127 属 rank7 分片 → 垃圾值）
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        from safetensors import safe_open
        wmap = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]

        def load_ref(name):
            with safe_open(f"{MODEL_DIR}/{wmap[name]}", framework="pt", device="cpu") as sf:
                return sf.get_tensor(name).float()

        def ref_gu(e):
            return torch.cat([load_ref(f"model.layers.0.mlp.experts.{e}.gate_proj.weight"),
                              load_ref(f"model.layers.0.mlp.experts.{e}.up_proj.weight")], dim=0)

        ref0, ref127 = ref_gu(0), ref_gu(127)
        ref0_dn = load_ref("model.layers.0.mlp.experts.0.down_proj.weight")

        # 逐层参照：探针捕获的 wrapper 可能不是 layer0（A2 探针证据：cpu_fwd
        # 变错层值而 CPU 张量不可能被改写 → 实际 self≠wrapped[0]），maxdiff
        # 必须对捕获层自身的 safetensors 参照计算
        _ref_cache: dict = {}

        def ref_gu_for(k, e):
            key = (k, e)
            if key not in _ref_cache:
                _ref_cache[key] = torch.cat(
                    [load_ref(f"model.layers.{k}.mlp.experts.{e}.gate_proj.weight"),
                     load_ref(f"model.layers.{k}.mlp.experts.{e}.up_proj.weight")], dim=0)
            return _ref_cache[key]

        w_gu = wrapped[0].experts.gate_up_proj         # [E, 2i, h] bf16 分片
        w_dn = wrapped[0].experts.down_proj
        with GatheredParameters([w_gu, w_dn]):
            t = w_gu.detach().float()
            d0 = float((t[0] - ref0.to(dev)).abs().max())
            d127 = float((t[127] - ref127.to(dev)).abs().max())
            ddn = float((w_dn.detach().float()[0] - ref0_dn.to(dev)).abs().max())
            row0 = t[0, 0, :4].tolist()
        d = torch.tensor([d0, d127, ddn], device=dev)
        dist.all_reduce(d, op=dist.ReduceOp.MAX)       # 跨 rank 一致性
        if rank == 0:
            print(f"[checksum:manual-gather] e0 gu maxdiff={float(d[0]):.6g} "
                  f"e127 gu maxdiff={float(d[1]):.6g} e0 dn maxdiff={float(d[2]):.6g}\n"
                  f"  gu[0] row0[:4]={row0}\n"
                  f"  file  row0[:4]={ref0[0, :4].tolist()}", flush=True)
        del d, t

        # 实验1（forward 前、无 hook 干扰）：与 GatheredParameters 完全同源的手动
        # 重组对拍。区分"重组逻辑 bug"与"forward 期间 ds_tensor 被引擎机制破坏"：
        # 若此处 maxdiff=0 而 eff-fwd 错 → 时机问题（hook/通信冲突）；
        # 若此处也错 → 重组逻辑 bug（结合 ds_tensor 原始内容定位）。
        from npuslim.qad.ste_experts import _zero3_gather_full
        p = wrapped[0].experts.gate_up_proj
        m = _zero3_gather_full(p).float()
        d_manual = float((m[0] - ref0.to(dev)).abs().max())
        del m
        if rank == 0:
            print(f"[checksum:manual-tensor] e0 maxdiff={d_manual:.6g}", flush=True)
            print(f"  ds_tensor shape={tuple(p.ds_tensor.shape)} "
                  f"dtype={p.ds_tensor.dtype} device={p.ds_tensor.device} "
                  f"ds_numel={p.ds_numel} ds_shape={tuple(p.ds_shape)}", flush=True)
            print(f"  ds[0:4]={p.ds_tensor[:4].float().flatten().tolist()}", flush=True)
            print(f"  file flat[0:4]={ref0.contiguous().view(-1)[:4].tolist()}", flush=True)
        dist.barrier()

        # 快照内容自检（方案 A2）：clone 时点在 initialize 后、任何 forward 前，
        # 探针已证明该时点 ds_tensor 内容==file；此处验证 NPU clone 忠实 + CPU
        # 快照忠实，并记录 id/ptr 供 forward 中对比（定位"对象被换"vs"内存被写"）
        if rank == 0:
            snap = wrapped[0]._ds_snapshot.get("gate_up")
            csnap = wrapped[0]._ds_cpu.get("gate_up")
            if snap is not None:
                print(f"  snap_gu[0:4]={snap[:4].float().flatten().tolist()} "
                      f"shape={tuple(snap.shape)} device={snap.device} "
                      f"id={id(snap)} ptr={snap.data_ptr()}", flush=True)
            if csnap is not None:
                print(f"  cpu_gu[0:4]={csnap[:4].tolist()} pinned={csnap.is_pinned()}",
                      flush=True)
        dist.barrier()

        # hook 侦察：找出 DeepSpeed 挂在 STE 包装器子树上的 forward hooks
        if rank == 0:
            for name, mod in wrapped[0].named_modules():
                ph = [repr(h)[:90] for h in mod._forward_pre_hooks.values()]
                fh = [repr(h)[:90] for h in mod._forward_hooks.values()]
                if ph or fh:
                    print(f"  [hooks] {name or '<STE>'}: pre={ph} post={fh}", flush=True)
        dist.barrier()

        orig_eff = type(wrapped[0])._effective_all
        seen: dict = {}
        events: list = []          # 前 16 次 _effective_all 调用序列 (layer_idx, proj)

        def _probe_eff(self, proj):
            out = orig_eff(self, proj)
            if len(events) < 16:
                events.append((getattr(self, "_qad_idx", -1), proj))
            if proj == "gate_up" and "gu" not in seen:
                # 捕获修复路径实际使用的有效权重，对拍【捕获层自身】的
                # safetensors 参照。ids：对比自检时点 → 定位对象是否被换；
                # ds_now：该 wrapper 的 ds_tensor 在 forward 中的逻辑头部
                t = out.detach().float()
                k = getattr(self, "_qad_idx", -1)
                r0, r127 = ref_gu_for(k, 0), ref_gu_for(k, 127)
                seen["gu"] = {
                    "idx": k, "wrap_id": id(self),
                    "cpu_id": id(self._ds_cpu.get("gate_up")),
                    "npu_id": id(self._ds_snapshot.get("gate_up")),
                    "d0": float((t[0] - r0.to(t.device)).abs().max()),
                    "d127": float((t[127] - r127.to(t.device)).abs().max()),
                    "e0": t[0, 0, :4].tolist(), "e127": t[127, 0, :4].tolist(),
                    "branch": ("cpu" if self._ds_cpu.get("gate_up") is not None
                               else "snap" if self._ds_snapshot.get("gate_up") is not None
                               else "fallback"),
                    "npu_ptr": (self._ds_snapshot["gate_up"].data_ptr()
                                if self._ds_snapshot.get("gate_up") is not None else None),
                    "cpu_fwd": (self._ds_cpu["gate_up"][:4].flatten().tolist()
                                if self._ds_cpu.get("gate_up") is not None else None),
                    "ds_now": (self.experts.gate_up_proj.ds_tensor[:4].float().flatten().tolist()
                               if getattr(self.experts.gate_up_proj, "ds_tensor", None) is not None
                               else None),
                }
            return out

        type(wrapped[0])._effective_all = _probe_eff
        inp0 = torch.from_numpy(first[0]).unsqueeze(0).to(dev)
        # forward 前最后一次快照复检：紧邻 engine() 调用，界定改写窗口
        if rank == 0:
            snap = wrapped[0]._ds_snapshot.get("gate_up")
            csnap = wrapped[0]._ds_cpu.get("gate_up")
            print(f"  [pre-fwd] npu_id={id(snap)} npu_ptr={snap.data_ptr()} "
                  f"npu[0:4]={snap[:4].float().flatten().tolist()} "
                  f"cpu[0:4]={csnap[:4].tolist()}", flush=True)
        dist.barrier()
        with torch.no_grad():
            engine(input_ids=inp0, use_cache=False)
        s = seen["gu"]
        if rank == 0:
            print(f"[checksum:eff-fwd#1] idx={s['idx']} branch={s['branch']} "
                  f"e0 maxdiff={s['d0']:.6g} e127 maxdiff={s['d127']:.6g}", flush=True)
            print(f"  wrap_id={s['wrap_id']} cpu_id={s['cpu_id']} npu_id={s['npu_id']} "
                  f"npu_ptr={s['npu_ptr']}", flush=True)
            print(f"  cpu_fwd[0:4]={s['cpu_fwd']} ds_now[0:4]={s['ds_now']}", flush=True)
            print(f"  [events]{events}", flush=True)
        events.clear()
        seen.clear()

        # 实验1g：训练同款 grad 上下文（checkpoint 内层实为 no_grad 执行，此处
        # 验证引擎在 grad 模式下 hook 行为是否不同）——skip-fq step0 loss 与
        # 修复前逐位一致（13.6097），需确认 grad 模式下快照路径是否被绕开
        engine(input_ids=inp0, use_cache=False)
        s1g = seen["gu"]
        if rank == 0:
            print(f"[checksum:eff-fwd#1g-grad] idx={s1g['idx']} branch={s1g['branch']} "
                  f"e0 maxdiff={s1g['d0']:.6g} e127 maxdiff={s1g['d127']:.6g}", flush=True)
            print(f"  cpu_fwd[0:4]={s1g['cpu_fwd']} ds_now[0:4]={s1g['ds_now']}", flush=True)
            print(f"  [events]{events}", flush=True)
        events.clear()
        seen.clear()
        del s1g

        # 实验2：移除全部 STE 包装器子树上的 DS hooks（pre/post fwd、bwd），
        # 消除引擎 gather/通信与 STE 内 ds_tensor all_gather 的交错，
        # 再对拍——若 maxdiff=0，则"移除 hooks + 手动 gather"为最终修复路径。
        _DS_HOOK_PATS = ("DeepSpeedZeRoOffload", "instrument_w_nvtx")

        def strip_ds_hooks(root):
            n = 0
            regs = ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks",
                    "_forward_pre_hooks_with_kwargs", "_forward_hooks_with_kwargs",
                    "_backward_pre_hooks", "_backward_pre_hooks_with_kwargs")
            for mod in root.modules():
                for reg in regs:
                    d = getattr(mod, reg, None)
                    if not d:
                        continue
                    for k in list(d.keys()):
                        if any(p in repr(d[k]) for p in _DS_HOOK_PATS):
                            del d[k]
                            n += 1
            return n

        n_removed = sum(strip_ds_hooks(w) for w in wrapped)
        if rank == 0:
            print(f"[strip-hooks] removed {n_removed} DS hooks from {len(wrapped)} STE subtrees",
                  flush=True)
        # strip 后 LoRA（已被 ZeRO 分片化）无 hook 无法 gather，einsum 会崩；
        # skip-fq 下 LoRA delta≡0（B=0），故 #2 直接返回 ds_tensor 重组的专家
        # 权重（等价于 orig_eff 输出），专拍"strip 后 forward 中重组正确性"
        def _probe_eff2(self, proj):
            w = self.experts.gate_up_proj if proj == "gate_up" else self.experts.down_proj
            if proj == "gate_up" and "gu" not in seen:
                # 流序判定探针：all_gather 后立即读（采样1）vs torch.npu.synchronize
                # 后重读同一缓冲（采样2）。采样1错+采样2对 → HCCL 完成对 compute
                # stream 不可见（cat/后续 op 读到分配器复用块旧内容=错层来源）；
                # 两次都错 → forward 期间 ds_tensor 源数据本身错。
                shard = w.ds_tensor.contiguous()
                parts = [torch.empty_like(shard)
                         for _ in range(dist.get_world_size())]
                dist.all_gather(parts, shard)
                rec = {
                    "ds_pre": w.ds_tensor[:4].float().flatten().tolist(),
                    "ptr_pre": w.ds_tensor.data_ptr(),
                    "p0_now": parts[0][:4].float().flatten().tolist(),
                }
                full = torch.cat(parts, dim=0)[:w.ds_numel]
                rec["flat4_now"] = full[:4].float().flatten().tolist()
                full_v = full.view(list(w.ds_shape)).to(torch.bfloat16)
                t = full_v.float()
                rec.update({
                    "d0": float((t[0] - ref0.to(t.device)).abs().max()),
                    "d127": float((t[127] - ref127.to(t.device)).abs().max()),
                    "e0": t[0, 0, :4].tolist(),
                    "m0": float(t[0].abs().mean()),
                })
                torch.npu.synchronize()
                rec.update({
                    "p0_sync": parts[0][:4].float().flatten().tolist(),
                    "flat4_sync": full[:4].float().flatten().tolist(),
                    "ds_post": w.ds_tensor[:4].float().flatten().tolist(),
                    "ptr_post": w.ds_tensor.data_ptr(),
                    "flat4_file": ref0.contiguous().view(-1)[:4].tolist(),
                })
                seen["gu"] = rec
                return full_v
            return _zero3_gather_full(w)

        type(wrapped[0])._effective_all = _probe_eff2
        with torch.no_grad():
            engine(input_ids=inp0, use_cache=False)
        s2 = seen["gu"]
        if rank == 0:
            print(f"[checksum:eff-fwd#2] e0 maxdiff={s2['d0']:.6g} "
                  f"e127 maxdiff={s2['d127']:.6g}", flush=True)
            print(f"  e  0 row0[:4]={s2['e0']} absmean={s2['m0']:.6g}", flush=True)
            print(f"  file  e0 row0[:4]={ref0[0, :4].tolist()}", flush=True)
            print(f"  ds_pre [0:4]={s2['ds_pre']} ptr={s2['ptr_pre']}", flush=True)
            print(f"  p0_now [0:4]={s2['p0_now']}  flat4_now={s2['flat4_now']}", flush=True)
            print(f"  p0_sync[0:4]={s2['p0_sync']} flat4_sync={s2['flat4_sync']}", flush=True)
            print(f"  ds_post[0:4]={s2['ds_post']} ptr={s2['ptr_post']}", flush=True)
            print(f"  file flat[0:4]={s2['flat4_file']}", flush=True)
        dist.barrier()

        # 实验3（ds_tensor 污染时序判定）：forward 全部结束后，用与 forward 前
        # 完全相同的重组函数再对拍一次。
        # - 若 maxdiff 变为 0.41x：引擎把错层数据写回 ds_tensor（永久污染）。
        # - 若仍为 0：forward 中为瞬时状态问题（结合 eff-fwd#2 的 now/sync 采样
        #   判定流序 vs 源数据）。
        m3 = _zero3_gather_full(p).float()
        d3 = float((m3[0] - ref0.to(dev)).abs().max())
        del m3
        if rank == 0:
            print(f"[checksum:post-fwd-manual] e0 maxdiff={d3:.6g}", flush=True)
            print(f"  ds_post_fwd[0:4]={p.ds_tensor[:4].float().flatten().tolist()} "
                  f"ptr={p.ds_tensor.data_ptr()}", flush=True)
            print(f"  file flat[0:4]={ref0.contiguous().view(-1)[:4].tolist()}", flush=True)
        dist.barrier()
        sys.exit(0)

    losses = []
    t_begin = time.time()
    total_tok = 0
    n_exec = 0   # 本 run 消费的步数（含 NaN 跳过）：final ckpt 标签，全 rank 一致
    for step in range(start_step, args.steps):
        t_fetch = time.time()
        inp_np, ids_np, lps_np = next(gen)   # 流式取样：永不耗尽，无 early-break 死锁
        n_exec = step + 1 - start_step
        ft_ms = (time.time() - t_fetch) * 1000
        inp = torch.from_numpy(inp_np).unsqueeze(0).to(dev)
        if step == 0:
            torch.npu.reset_peak_memory_stats(local_rank)
        # use_cache=False：DynamicCache 会被 checkpoint 重算二次 update（49×512 vs
        # 48×512）导致重算元数据校验失败，且徒增 cache 显存
        out = engine(input_ids=inp, use_cache=False)
        lg = out.logits[0, :ids_np.shape[0]].float()   # 行 i ↔ teacher 行 i（i=0..L_use-1）
        if step == 0:
            mem_report("forward后")
        ids_t = torch.from_numpy(ids_np).long().to(dev)
        lps_t = torch.from_numpy(lps_np).float().to(dev)
        loss = kl_distill_loss(lg, ids_t, lps_t, VOCAB, tau=1.0, chunk=VCHUNK)
        # NaN/inf 不能单 rank skip：其余 rank 继续 backward/step，梯度 reduce-scatter
        # 等待本 rank → 集合不对称死锁（与 2026-09-08 样本耗尽死锁同机理）。
        # 全 rank 投票后共同跳过，保持集合通信对称
        ok = torch.isfinite(loss).to(torch.float32)
        dist.all_reduce(ok, op=dist.ReduceOp.SUM)
        if ok.item() < world:
            if rank == 0:
                print(f"  step {step}: NaN/inf loss，全 rank 共同跳过", flush=True)
            engine.zero_grad()
            continue
        engine.zero_grad()
        engine.backward(loss)
        if step == 0:
            mem_report("backward后")
        engine.step()
        loss_v = float(loss.item())
        losses.append(loss_v)
        total_tok += ids_np.shape[0]
        if step % 5 == 0 or step == args.steps - 1:
            loss_t = torch.tensor(loss_v, device=dev)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            if rank == 0:
                gn = engine.get_global_grad_norm()
                gn = f"{float(gn):.3f}" if gn is not None else "n/a"
                print(f"  step {step:3d}: loss={float(loss_t):.4f} grad_norm={gn} "
                      f"fetch={ft_ms:.0f}ms", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(step + 1, "periodic")
        del out, lg
        torch.npu.empty_cache()

    dist.barrier()
    if args.save_every:
        save_ckpt(start_step + n_exec, "final")   # 含 NaN 跳过情况；n_exec 全 rank 一致
    # 全局吞吐口径（此前只统计 rank0 自身，"8 rank"标注有误导）：all_reduce 求和
    # 注意 HCCL 不支持 float64 all_reduce（ERR02007），用 float32（token 数 < 2^24 精确）
    t_tok = torch.tensor([total_tok], dtype=torch.float32, device=dev)
    dist.all_reduce(t_tok, op=dist.ReduceOp.SUM)
    total_tok_all = int(t_tok.item())
    if rank == 0:
        dt = time.time() - t_begin
        tps = total_tok_all / dt
        print("=== 结果 ===")
        print(f"  tokens/s: {tps:.1f}（全局 {total_tok_all} tokens / {dt:.0f}s，{world} rank）")
        print(f"  外推 100M tokens ≈ {100_000_000/tps/3600:.1f} h（1 epoch）")
        print(f"  loss 序列: {[f'{x:.3f}' for x in losses[:12]]}...")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
