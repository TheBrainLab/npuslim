#!/bin/bash -li
# ds_train_tiny_glm.py — GLM QAD tiny e2e 预演训练器（ds_train_t7.py 的净版 GLM 适配）
#
# 差异（相对 ds_train_t7.py，M4/Qwen3 版）：
#   * MODEL_DIR/CACHE/TRAIN 重指向 tiny_glm5 / logits_cache_tiny / tiny_train_v0
#   * encode_one 窗口 8192 → 1024（与 tiny teacher 缓存构建参数一致，token 流对齐）
#   * wrap_all_experts → GLM 变体：跳过 dense MLP 层（GlmMoeDsaMLP 无 .experts）
#   * 删除 M4 ZeRO-3 排障期的 checksum/探针块（395-620），保留正式训练路径
#   * ZeRO-3 world=1（单卡；保留引擎栈保真度：ds_tensor 快照方案 A2 照常生效）
#
# 启动（d_series_env，卡 3）:
#   torchrun --nproc_per_node=1 ds_train_tiny_glm.py --steps 20 --save-every 20
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

sys.path.insert(0, f"{SHARED}/ylt/npuslim/src")
sys.path.insert(0, f"{SHARED}/ylt/glm52-qad/precompute")

from npuslim.qad.teacher_cache import TeacherCache  # noqa: E402
from npuslim.qad.kl_loss import kl_distill_loss    # noqa: E402
from npuslim.qad.ste_experts import STEQwen3Experts  # noqa: E402
from precompute_logits import encode_one            # noqa: E402

MODEL_DIR = os.environ.get("C3_MODEL_DIR", f"{SHARED}/ylt/npuslim/workspace/tiny_glm5")
CACHE = Path(f"{SHARED}/ylt/glm52-qad/precompute/logits_cache_tiny")
TRAIN = Path(f"{SHARED}/ylt/glm52-qad/qad_data/tiny_train_v0.jsonl")
CORPUS = Path(f"{SHARED}/ylt/glm52-qad/glm52-qad-corpus/"
              "processed/corpus_v1_2026-08-27/clean")
VOCAB = 154_880     # tiny_glm5 config.vocab_size
VCHUNK = 65536


def build_ds_config(world: int, lr: float) -> dict:
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "leaf_module": {"classes": []},   # GLM 同理：禁用默认 leaf hook
            "stage3_param_persistence_threshold": 25_000_000,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        },
        "optimizer": {"type": "AdamW",
                      "params": {"lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.0}},
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }


def wrap_all_experts_glm(model, rank: int, vectorized: bool = True,
                         dtype=torch.bfloat16, zero3_gather: bool = True):
    """GLM 变体：MoE 层 mlp.experts（GlmMoeDsaNaiveMoe，属性与 Qwen3MoeExperts
    同构：num_experts/hidden_dim/intermediate_dim/act_fn/gate_up_proj/down_proj）
    直接复用 STEQwen3Experts；dense MLP 层（GlmMoeDsaMLP，无 .experts）跳过。"""
    wrapped = []
    for li, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue                      # dense MLP 层（first_k_dense_replace）
        exp = layer.mlp.experts
        for p in exp.parameters():
            p.requires_grad = False
        w = STEQwen3Experts(exp, rank=rank, vectorized=vectorized, dtype=dtype,
                            zero3_gather=zero3_gather)
        w.to(exp.gate_up_proj.device)
        layer.mlp.experts = w
        wrapped.append(w)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--micro-tokens", type=int, default=512)
    ap.add_argument("--min-enc-len", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)     # LoRA rank
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--save-dir", default="ckpt_tiny_glm")
    ap.add_argument("--skip-fake-quant", action="store_true",
                    help="正确性锚点：禁用 fake-quant（LoRA B=0 → 学生=教师），"
                         "loss 应≈0（±bf16/尾近似噪声）；用于判定 STE 路径真实生效")
    args = ap.parse_args()

    if args.skip_fake_quant:
        import npuslim.qad.ste_experts as _ste_exp
        _ste_exp.fake_quant_weight = lambda w, bits=4, scale=None: w
        if int(os.environ.get("RANK", 0)) == 0:
            print("  [anchor] fake-quant 已禁用（学生=教师，loss 应≈0）", flush=True)

    dist.init_process_group(backend="hccl",
                            timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.npu.set_device(local_rank)
    dev = f"npu:{local_rank}"
    if rank == 0:
        print(f"DS-ZeRO3 world={world} micro={args.micro_tokens} "
              f"total/step={world * args.micro_tokens}", flush=True)

    # ---- 数据管线：与 M4 同构（TeacherCache + 流式 epoch 生成器） ----
    tc = TeacherCache(CACHE)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t0 = time.time()
    entries = []
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
    n_need = max(args.steps, 4)
    pool = list(mine)
    if len(pool) > n_need * 2:
        stride = max(1, len(pool) * 10 // (n_need * 13))
        pool = pool[::stride]
    order = {sid: (tc._by_id[sid][-1].shard, tc._by_id[sid][-1].row)
             for _, sid in pool if sid in tc._by_id}
    pool.sort(key=lambda e: order.get(e[1], ("", -1)))
    line_cache: dict = {}

    BLOCK = 16

    def fetch_stream():
        n_pool = len(pool)
        brng = random.Random(1337)
        epoch = 0
        while True:
            off = (epoch * 37) % max(n_pool, 1)
            for b0 in range(0, max(n_pool, 1), BLOCK):
                block = [pool[(i + off) % n_pool] for i in range(b0, min(b0 + BLOCK, n_pool))]
                brng.shuffle(block)
                for key, sid in block:
                    fname, _, ln = key.rpartition(":")
                    if fname not in line_cache:
                        with open(CORPUS / fname, encoding="utf-8") as fh:
                            line_cache[fname] = fh.readlines()
                    row = json.loads(line_cache[fname][int(ln)].strip())
                    enc = encode_one(tok, row, 1024)   # 与 tiny 缓存构建口径一致
                    if not (enc is not None and not isinstance(enc, Exception) and len(enc) > 1):
                        continue
                    if len(enc) < args.min_enc_len:
                        continue
                    expected = np.asarray(enc[1:], dtype=np.int32)
                    if expected.shape[0] < 8:
                        continue
                    fs = tc.fetch(sid, expected_ids=expected)
                    L_enc = min(len(enc), args.micro_tokens)
                    L_use = min(fs.topk_ids.shape[0], L_enc - 1)
                    yield (np.asarray(enc, dtype=np.int64)[:L_enc],
                           fs.topk_ids[:L_use].copy(), fs.topk_lps[:L_use].copy())
            epoch += 1

    gen = fetch_stream()
    first = next(gen)
    dt = time.time() - t0
    print(f"  [rank{rank}] 流式取样就绪 pool={len(pool)} 首样本 {dt:.1f}s", flush=True)

    from transformers import AutoModelForCausalLM
    from transformers.integrations import HfDeepSpeedConfig
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    ds_config = build_ds_config(world, args.lr)
    dschf = HfDeepSpeedConfig(ds_config)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    if rank == 0:
        print(f"  zero.Init 分片加载 {time.time()-t0:.0f}s", flush=True)

    wrapped = wrap_all_experts_glm(model, rank=args.rank, dtype=torch.bfloat16,
                                   zero3_gather=True)
    for i, w in enumerate(wrapped):
        w._qad_idx = i
    for p in model.parameters():
        p.requires_grad = False
    for w in wrapped:
        for p in w.lora.params.values():
            p.requires_grad = True
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    if rank == 0:
        print(f"  GLM MoE 层 STE 包装 {len(wrapped)} 个（dense 跳过）；可训练 {n_train/1e6:.1f}M",
              flush=True)

    def _make_ckpt_fwd(orig_fwd):
        def _ckpt_fwd(*a, **kw):
            return torch.utils.checkpoint.checkpoint(
                orig_fwd, *a, use_reentrant=False, **kw)
        return _ckpt_fwd

    for layer in model.model.layers:
        layer.forward = _make_ckpt_fwd(layer.forward)

    opt = torch.optim.AdamW(train_params, lr=args.lr)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)
    if rank == 0:
        part = sum(1 for p in model.parameters()
                   if getattr(p, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE)
        persist = sum(1 for p in model.parameters()
                      if getattr(p, "ds_status", None) == ZeroParamStatus.AVAILABLE)
        normal = sum(1 for p in model.parameters()
                     if getattr(p, "ds_status", None) is None)
        print(f"  参数状态: 分片{part} / 常驻{persist} / 普通 LoRA{normal}", flush=True)

    # 方案 A2：engine.initialize 后、首次 forward 前的私有快照（对照 M4 沉淀）
    for w in wrapped:
        w.snapshot_ds_tensors()
    if rank == 0:
        n_snap = sum(len(w._ds_snapshot) for w in wrapped)
        print(f"  [fix-A] 专家参数 ds_tensor 私有快照 {n_snap}/{2 * len(wrapped)}", flush=True)

    if os.environ.get("STRIP_DS_HOOKS") == "1":
        # 方案 B A/B 验证（§3.5-C1）：剥离 STE 专家模块的 DS forward 期 hook
        # （pre=gather / post=释放）——STE 经 A2 CPU 快照自管参数，tiny 上对比
        # STRIP 前后的 loss 一致性即正确性验证。
        n_strip = 0
        for w in wrapped:
            m = w.experts
            for hook_dict in (m._forward_pre_hooks, m._forward_hooks):
                for key in list(hook_dict):
                    h = hook_dict[key]
                    qual = getattr(h, "__qualname__", "") or type(h).__name__
                    if "deepspeed" in qual.lower():
                        del hook_dict[key]
                        n_strip += 1
        if rank == 0:
            print(f"  [strip] DS expert-module hooks removed: {n_strip}", flush=True)
            for name, mod in wrapped[0].named_modules():
                hs = {**{f"pre{k}": type(h).__name__ + ":" + str(getattr(h, "__qualname__", ""))[:50]
                         for k, h in mod._forward_pre_hooks.items()},
                      **{f"post{k}": type(h).__name__ for k, h in mod._forward_hooks.items()}}
                if hs:
                    print(f"  [strip-dbg] {name or '<wrapper>'}: {hs}", flush=True)

    lora_state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            lora_state[n] = p
    if args.save_every:
        os.makedirs(args.save_dir, exist_ok=True)

    def save_ckpt(step_done: int, tag: str):
        state = {}
        for n, p in lora_state.items():
            state[n] = p.detach().to(torch.bfloat16).contiguous().cpu()
        torch.save(state, f"{args.save_dir}/lora_step{step_done}_{tag}.pt")
        if rank == 0:
            print(f"  saved {args.save_dir}/lora_step{step_done}_{tag}.pt", flush=True)

    losses = []
    t_begin = time.time()
    total_tok = 0
    n_exec = 0
    start_step = 0
    first[0].shape[0]   # 维度自检（防 silent shape 错配）
    for step in range(start_step, args.steps):
        t_fetch = time.time()
        inp_np, ids_np, lps_np = next(gen)
        n_exec = step + 1 - start_step
        ft_ms = (time.time() - t_fetch) * 1000
        inp = torch.from_numpy(inp_np).unsqueeze(0).to(dev)
        out = engine(input_ids=inp, use_cache=False)
        lg = out.logits[0, :ids_np.shape[0]].float()
        ids_t = torch.from_numpy(ids_np).long().to(dev)
        lps_t = torch.from_numpy(lps_np).float().to(dev)
        loss = kl_distill_loss(lg, ids_t, lps_t, VOCAB, tau=1.0, chunk=VCHUNK)
        ok = torch.isfinite(loss).to(torch.float32)
        dist.all_reduce(ok, op=dist.ReduceOp.SUM)
        if ok.item() < world:
            if rank == 0:
                print(f"  step {step}: NaN/inf loss，全 rank 共同跳过", flush=True)
            engine.zero_grad()
            continue
        engine.zero_grad()
        engine.backward(loss)
        engine.step()
        loss_v = float(loss.item())
        losses.append(loss_v)
        total_tok += ids_np.shape[0]
        if step % 2 == 0 or step == args.steps - 1:
            gn = engine.get_global_grad_norm()
            gn = f"{float(gn):.3f}" if gn is not None else "n/a"
            if rank == 0:
                print(f"  step {step:3d}: loss={loss_v:.4f} grad_norm={gn} "
                      f"fetch={ft_ms:.0f}ms", flush=True)
        del out, lg
        torch.npu.empty_cache()

    dist.barrier()
    if args.save_every:
        save_ckpt(start_step + n_exec, "final")
    if rank == 0:
        dt = time.time() - t_begin
        print("=== tiny e2e 结果 ===")
        print(f"  loss 序列: {[f'{x:.3f}' for x in losses]}")
        print(f"  tokens/s: {total_tok / dt:.1f}（{total_tok} tokens / {dt:.0f}s）")
        print(f"  对照锚点：skip-fq/B=0 时 loss 应≈0；此 run（LoRA 学补偿）loss 从"
              f"量化创伤水平起步——数值健康 = STE/teacher/DSA 前向三件套打通")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
