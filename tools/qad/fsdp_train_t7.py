"""M4 FSDP 冒烟训练（8 NPU）：专家层 FSDP 分片 + STE + LoRA + chunked KL。

并行设计：
- 仅 MoE 专家层（STE 包装器，含 29B 权重 + LoRA）FSDP 分片到 8 卡
- attention/norm/embed/lm_head（~1.5B）每 rank 复制（内存充裕）
- 数据并行：每 rank 独立 micro-batch，总 batch = 8 × micro-tokens

用法（torchrun 8 进程）：
    python -m torch.distributed.run --nproc_per_node=8 fsdp_train_t7.py
        [--steps N] [--micro-tokens L] [--rank R] [--lr L]
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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, f"{SHARED_HOST}/ylt/npuslim/src")
sys.path.insert(0, f"{SHARED_HOST}/ylt/qwen3-qad/precompute")

from npuslim.qad.teacher_cache import TeacherCache, FetchedSample  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro-tokens", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    dist.init_process_group(backend="hccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.npu.set_device(local_rank)
    dev = f"npu:{local_rank}"
    if rank == 0:
        print(f"FSDP world={world} micro={args.micro_tokens} total/step={world*args.micro_tokens}")

    # ---- 数据：每 rank 独立取 micro_tokens 行（不同样本） ----
    tc = TeacherCache(CACHE)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    line_cache: dict[str, list[str]] = {}
    fetched: list[FetchedSample] = []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s["category"] not in ("agentic_coding", "general"):
                continue
            fname, _, ln = s["sample_key"].rpartition(":")
            if fname not in line_cache:
                with open(CORPUS / fname, encoding="utf-8") as fh:
                    line_cache[fname] = fh.readlines()
            row = json.loads(line_cache[fname][int(ln)].strip())
            enc = encode_one(tok, row, 8192)
            if not (enc is not None and not isinstance(enc, Exception) and len(enc) > 1):
                continue
            expected = np.asarray(enc[1:], dtype=np.int32)
            if expected.shape[0] < 20:
                continue
            fs = tc.fetch(s["id"], expected_ids=expected)
            L = min(fs.actual_ids.shape[0], args.micro_tokens)
            fetched.append(FetchedSample(sample_id=fs.sample_id, actual_ids=fs.actual_ids[:L],
                                         topk_ids=fs.topk_ids[:L], topk_lps=fs.topk_lps[:L]))
            if len(fetched) >= 3 * world:
                break
    # 每 rank 轮流取：rank r 用 fetched[r], fetched[r+world], ...
    if len(fetched) < world:
        if rank == 0:
            print("ERROR: 样本不足")
        dist.barrier()
        sys.exit(2)
    print(f"  [rank{rank}] 样本数 {len(fetched)}")

    # ---- 模型加载（CPU）→ STE 包装 → 冻结 → 标准全模型 FSDP（auto_wrap_policy） ----
    from transformers import AutoModelForCausalLM
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    # STE + LoRA 包装（层级 checkpoint 承担激活重算，STE 层关掉避免重复）
    wrapped = wrap_all_experts(model, rank=args.rank, vectorized=True,
                               dtype=torch.bfloat16, use_checkpoint=False)
    for p in model.parameters():
        p.requires_grad = False
    for w in wrapped:
        for p in w.lora.params.values():
            p.requires_grad = True
    # 可训练参数必须在 FSDP 包装前收集（use_orig_params=True 保留原 param 对象）
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    # 层级 activation checkpointing：backward 重算层前向 → FSDP 前向 gather 的权重
    # 不再被 autograd graph 引用，可逐层释放（绕开 torch_npu FSDP gather 累积）
    for layer in model.model.layers:
        orig_fwd = layer.forward

        def _ckpt_fwd(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                orig_fwd, *args, use_reentrant=False, **kwargs)
        layer.forward = _ckpt_fwd
    # 标准 FSDP：ModuleWrapPolicy 以 decoder layer 为 FSDP 单元包整个模型
    policy = ModuleWrapPolicy({Qwen3MoeDecoderLayer})
    model = FSDP(model, device_id=local_rank, use_orig_params=True, auto_wrap_policy=policy)
    if rank == 0:
        print(f"  加载+包装 {time.time()-t0:.0f}s，标准 auto_wrap FSDP 完成，"
              f"可训练 {n_train/1e6:.1f}M")

    opt = torch.optim.AdamW(train_params, lr=args.lr)

    def mem_report(tag):
        a = torch.npu.memory_allocated(local_rank) / 2**30
        p = torch.npu.max_memory_allocated(local_rank) / 2**30
        print(f"  [mem:{tag} rank{rank}] alloc={a:.1f}GiB peak={p:.1f}GiB", flush=True)
        dist.barrier()

    # ---- 训练循环 ----
    losses = []
    t_begin = time.time()
    total_tok = 0
    for step in range(args.steps):
        fs = fetched[step % len(fetched)]
        inp = torch.from_numpy(fs.actual_ids.astype(np.int64)).unsqueeze(0).to(dev)
        if step == 0:
            for d in range(world):
                torch.npu.reset_peak_memory_stats(d)
        out = model(input_ids=inp)
        lg = out.logits[0].float()
        if step == 0:
            mem_report("forward后")
        L = lg.shape[0]
        ids_t = torch.from_numpy(fs.topk_ids[:L]).long().to(dev)
        lps_t = torch.from_numpy(fs.topk_lps[:L]).float().to(dev)
        loss = kl_distill_loss(lg, ids_t, lps_t, VOCAB, tau=1.0, chunk=VCHUNK)
        if not torch.isfinite(loss):
            if rank == 0:
                print(f"  step {step}: NaN loss")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            mem_report("backward后")
        opt.step()
        loss_v = float(loss.item())
        losses.append(loss_v)
        total_tok += L
        # rank0 汇总
        if step % 5 == 0 or step == args.steps - 1:
            loss_t = torch.tensor(loss_v, device=dev)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"  step {step:3d}: loss={float(loss_t):.4f}")
        del out, lg
        torch.npu.empty_cache()

    dist.barrier()
    if rank == 0:
        dt = time.time() - t_begin
        tps = total_tok / dt
        print(f"=== 结果 ===")
        print(f"  tokens/s: {tps:.1f}（{total_tok} tokens / {dt:.0f}s，{world} rank）")
        print(f"  外推 100M tokens ≈ {100_000_000/tps/3600:.1f} h")
        print(f"  loss 序列: {[f'{x:.3f}' for x in losses[:12]]}...")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
