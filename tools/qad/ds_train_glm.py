#!/bin/bash -li
# ds_train_glm.py — GLM-5.2 全量 QAD 训练器（C2；ds_train_tiny_glm.py 的全量 GLM 版）
#
# 相对 ds_train_tiny_glm.py（tiny 预演）的差异：
#   * MODEL_DIR → GLM-5.2-bf16（全量 355B）
#   * CACHE → glm52-qad/precompute/logits_cache（GLM teacher 缓存，415G/550.5M tokens）
#   * TRAIN → glm52-qad/qad_data/qad_train_v1.jsonl（GLM sample_key 表）
#   * encode_one 窗口 → 8192（与 GLM teacher 缓存构建口径一致：--max_len 8192 --top_k 100）
#   * wrap_all_experts → npuslim.qad.ste_experts.wrap_all_experts_glm（C1 正式化）
#   * VOCAB = 154880（GLM-5.2 config.vocab_size）
#   * 全量 ZeRO-3 world=16（355B bf16≈710GB → 16 卡 = 44GB/卡；见 0908 §4.8 C4）
#
# 启动（16 卡节点对，torchrun --nproc_per_node=16）:
#   torchrun --nproc_per_node=16 ds_train_glm.py --steps N --micro-tokens L
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

sys.path.insert(0, f"{QAD_ROOT}/precompute")
sys.path.insert(0, QAD_ROOT)
sys.path.insert(0, NPUSLIM_SRC)
sys.path.insert(0, f"{YLT}/npuslim_repo/tools/qad")  # repo 单一源优先（diag_ep/prof_ste），运行时 glm52-qad 为回退

_QAD_PROFILE = os.environ.get("QAD_PROFILE") == "1"
if _QAD_PROFILE:
    import prof_ste  # noqa: E402  installs ste_experts monkeypatches as side effect
import diag_ep  # noqa: E402  QAD_DIAG=1 门控诊断（未设时严格 no-op）

# torch_npu profiler（QAD_TPROF=1 门控，仅 rank0）：真实 compute/comm/host-gap 分解。
# 注意：与 QAD_DIAG/QAD_PROFILE 的 sync 钩子互斥使用，否则分解被 sync 扭曲。
_TP_ON = os.environ.get("QAD_TPROF") == "1"
_TP_START = int(os.environ.get("QAD_TPROF_START", "6"))
_TP_ACTIVE = int(os.environ.get("QAD_TPROF_ACTIVE", "2"))
_TP_OUT = os.environ.get("QAD_TPROF_OUT", "tprof_r0.json")


def _tp_make():
    from torch_npu.profiler import ProfilerActivity, profile
    kwargs = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                  record_shapes=False, profile_memory=False, with_stack=False)
    try:  # Level1 提升 hccl 算子可见性；API 不在时退回默认（仍有算子级分解）
        from torch_npu.profiler import ProfilerLevel, _ExperimentalConfig
        kwargs["experimental_config"] = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1)
    except Exception as exc:  # noqa: BLE001 诊断能力降级，不阻塞训练
        print(f"[tprof] experimental_config unavailable ({exc}); default level",
              flush=True)
    return profile(**kwargs)


def _tp_report(prof) -> None:
    try:
        prof.export_chrome_trace(_TP_OUT)
        rows = sorted(((getattr(e, "self_device_time_total", 0)
                        or getattr(e, "self_cuda_time_total", 0) or 0,
                        e.count, e.key) for e in prof.key_averages()), reverse=True)
        print(f"[tprof] trace -> {_TP_OUT}; top device ops:", flush=True)
        for dt, cnt, k in rows[:25]:
            print(f"  [tprof] {dt / 1e3:9.1f}ms x{cnt:<6d} {k[:110]}", flush=True)
    except Exception as exc:  # noqa: BLE001 诊断导出失败不阻塞训练
        print(f"[tprof] export failed: {exc}", flush=True)

from npuslim.qad.teacher_cache import TeacherCache  # noqa: E402
from npuslim.qad.kl_loss import kl_distill_loss    # noqa: E402
from npuslim.qad.ste_experts import STEQwen3Experts, wrap_all_experts_glm  # noqa: E402
from precompute_logits import encode_one            # noqa: E402

MODEL_DIR = os.environ.get("GLM_MODEL_DIR", f"{SHARED}/models/GLM-5.2-bf16/snapshots/master")
CACHE = Path(os.environ.get("GLM_CACHE",
    f"{SHARED}/ylt/glm52-qad/precompute/logits_cache"))   # GLM teacher 缓存（415G/550.5M tokens；--max_len 8192 --top_k 100）
TRAIN = Path(os.environ.get("GLM_TRAIN",
    f"{SHARED}/ylt/glm52-qad/qad_data/qad_train_v1.jsonl"))
ENC_WINDOW = int(os.environ.get("GLM_ENC_WINDOW", "8192"))   # 与 GLM teacher 缓存构建口径一致；tiny 冒烟可覆盖为 1024
CORPUS = Path(f"{SHARED}/ylt/glm52-qad/glm52-qad-corpus/"
              "processed/corpus_v1_2026-08-27/clean")
VOCAB = 154_880     # GLM-5.2 config.vocab_size
VCHUNK = 65536


def build_ds_config(world: int, lr: float) -> dict:
    cfg = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "leaf_module": {"classes": []},   # GLM 同理：禁用默认 leaf hook
            "stage3_param_persistence_threshold": int(
                os.environ.get("ZERO3_PERSIST_THRESHOLD", "25000000")),
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
    if os.environ.get("OFFLOAD_PARAM") == "1":
        cfg["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": False}
    return cfg


# wrap_all_experts_glm 已正式化进 npuslim.qad.ste_experts（C1，含单测
# tests/qad/test_glm_ste_experts.py），此处直接 import 使用（见文件头 import）。


def _read_peak_rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def _start_mem_watchdog(threshold_gib: int = 80, interval_s: int = 10) -> None:
    """★ OOM 预警（§6e-3④）：MemAvailable 低于阈值时自杀并报告原因。

    动机：C4-24 探针 24 rank 的 harness host 内存打穿 avail → 内核 oom-killer
    随机杀系统件（k8s-fuse-plugin 死 = dtfs 全线 IO error）→ 连锁故障。
    自杀是确定性的：只死自己，不拖累系统件。阈值默认 80 GiB（系统件 ~150 GiB
    + 余量），每 10s 采样一次（每 rank 独立采样，任一 rank 触发即整体退出——
    torchrun 会 teardown 其余 rank）。
    """
    import threading

    def _loop():
        while True:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            avail_gib = int(line.split()[1]) / 1048576
                            break
                if avail_gib < threshold_gib:
                    rank = int(os.environ.get("RANK", "-1"))
                    node = os.uname().nodename
                    print(f"[MEM-WATCHDOG] node={node} rank={rank} "
                          f"MemAvailable={avail_gib:.0f}GiB < {threshold_gib}GiB 阈值 "
                          f"→ 主动自杀（防 oom-killer 屠杀系统件/dtfs-fuse，§6e-3④）",
                          flush=True)
                    os._exit(42)
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


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
    ap.add_argument("--deploy-scale-dir", default=None,
                    help="P1 部署产物目录；给出则专家 fake-quant 用其 SSZ weight_scale 对齐"
                         "（QAD Phase-1），默认 GLM_DEPLOY_SCALE_DIR 环境变量")
    ap.add_argument("--opt-mode", choices=("adamw", "sgd_stub", "adamw_bf16"), default="adamw",
                    help="C4-24 probe optimizer: adamw(生产口径, fp32 m+v ~150GiB/rank) / "
                         "sgd_stub(最小探针) / adamw_bf16(bf16 m+v ~50GiB/rank 近生产体型)"),
    ap.add_argument("--init-mode", choices=("checkpoint", "random", "stream"), default="checkpoint",
                    help="random = 绕开 from_pretrained、在 zero.Init 下直建随机权重模型"
                         "（C4 内存探针用，规避 HF from_pretrained×定制 MoE 的加载路径 OOM）；"
                         "stream = zero.Init+from_config 建空模后用 stream_load_checkpoint 逐张量"
                         "流式灌入 safetensors（内存精瘦，S2 生产路径，规避 from_pretrained 3-5x 放大）")
    ap.add_argument("--skip-wrap", action="store_true",
                    help="隔离试验②：跳过 STE 包装/LoRA/快照，仅最后层 attention 2D 参数可训练，"
                         "直接 engine+step——区分 4 节点 HCCL 因素与 STE 包装因素（§6.7 C4-32 SIGSEGV）")
    ap.add_argument("--frozen-source", choices=("checkpoint", "artifact"), default="checkpoint",
                    help="checkpoint=当前行为（bf16 checkpoint）；artifact=B2（W4A8 部署产物反量化权重，"
                         "学生前向=已部署量化模型，LoRA 学纯补偿）。artifact 模式强制 fake-quant 恒等")
    ap.add_argument("--artifact-dir", default=None,
                    help="B2 frozen-source=artifact 的 W4A8 部署产物目录（含 model.safetensors.index.json）")
    args = ap.parse_args()
    # EP 路径的统一 token 容量（T ≤ micro-tokens 恒成立 → 零同步 all_to_all）
    os.environ["QAD_EP_TMAX"] = str(args.micro_tokens)
    # ★ OOM watchdog（§6e-3④）：阈值可 env 覆盖；放最前，全程守护
    _start_mem_watchdog(threshold_gib=int(os.environ.get("MEM_WATCHDOG_GIB", "80")),
                        interval_s=int(os.environ.get("MEM_WATCHDOG_INTERVAL", "10")))

    if args.frozen_source == "artifact" and args.artifact_dir is None:
        if int(os.environ.get("RANK", 0)) == 0:
            print("  [FATAL] --frozen-source artifact 需要 --artifact-dir", flush=True)
        os._exit(2)

    if args.frozen_source == "artifact" and not args.skip_fake_quant:
        args.skip_fake_quant = True
        if int(os.environ.get("RANK", 0)) == 0:
            print("  [B2] frozen-source=artifact → 强制 fake-quant 恒等"
                  "（基座已量化-反量化，LoRA 学纯补偿）", flush=True)

    if args.skip_fake_quant:
        import npuslim.qad.ste_experts as _ste_exp
        _ste_exp.fake_quant_weight = lambda w, bits=4, scale=None: w
        if int(os.environ.get("RANK", 0)) == 0:
            if args.frozen_source == "artifact":
                print("  [anchor-B2] fake-quant 已禁用（学生=反量化部署模型）。"
                      "注意：dequant 基座下 skip-fake-quant 不再意味着 loss≈0——"
                      "部署模型与 bf16 教师的差（量化创伤）正是 LoRA 要补偿的，"
                      "loss 应从创伤水平起步并下降", flush=True)
            else:
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
                    enc = encode_one(tok, row, ENC_WINDOW)   # 与 GLM teacher 缓存构建口径一致（--max_len 8192）
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
    t0 = time.time()
    if args.init_mode in ("random", "stream"):
        # random 与 stream 都在 zero.Init 下 from_config 建模（不物化 checkpoint，
        # 规避 from_pretrained 3-5x 主存放大）。random 留参数未初始化（探针用）；
        # stream 建模后用 stream_load_checkpoint 逐张量流式灌入 safetensors。
        import deepspeed as _deepspeed
        from transformers import AutoConfig, PreTrainedModel
        cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
        # zero.Init 包装 module.apply：post_init→init_weights 的 apply 遍历触发每层 zero 参数的
        # all_gather(全参数物化)+HCCL broadcast+重分片——GLM 融合专家参数单张 12.9GB，双份即破 64GB
        # （Qwen3 最大参数 <2GB 故 M4 未暴露）。探针不需真权重初始化 → 跳过该 apply 遍历。
        _orig_init_weights = PreTrainedModel.init_weights
        PreTrainedModel.init_weights = lambda self: None
        try:
            with _deepspeed.zero.Init(config_dict_or_path=ds_config,
                                      dtype=torch.bfloat16, enabled=True,
                                      remote_device="cpu", pin_memory=False):
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        finally:
            PreTrainedModel.init_weights = _orig_init_weights
        if args.init_mode == "stream":
            t_load = time.time()
            if args.frozen_source == "artifact":
                from stream_load_checkpoint import stream_load_artifact
                stream_load_artifact(model, args.artifact_dir, rank, world,
                                     verbose=True, log_fn=print)
                if rank == 0:
                    print(f"  [stream-artifact] W4A8 artifact dequant stream-loaded "
                          f"{time.time()-t_load:.0f}s", flush=True)
            else:
                from stream_load_checkpoint import stream_load_checkpoint
                stream_load_checkpoint(model, MODEL_DIR, rank, world, verbose=True, log_fn=print)
                if rank == 0:
                    print(f"  [stream] checkpoint stream-loaded {time.time()-t_load:.0f}s", flush=True)
        elif os.environ.get("C4_PROBE"):
            _rss = _read_peak_rss_gib()
            print(f"  [c4] random-init built {time.time()-t0:.0f}s | "
                  f"HBM alloc={torch.npu.memory_allocated()/2**30:.2f}GiB "
                  f"max={torch.npu.max_memory_allocated()/2**30:.2f}GiB | "
                  f"host RSS peak={_rss:.1f}GiB", flush=True)
    else:
        dschf = HfDeepSpeedConfig(ds_config)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
        if rank == 0:
            print(f"  zero.Init 分片加载 {time.time()-t0:.0f}s", flush=True)

    deploy_dir = args.deploy_scale_dir or os.environ.get("GLM_DEPLOY_SCALE_DIR")
    if args.frozen_source == "artifact" and not deploy_dir and args.artifact_dir:
        deploy_dir = args.artifact_dir
    scale_provider = None
    if deploy_dir and not args.skip_wrap:
        from npuslim.qad.deploy_scales import make_scale_provider
        n_exp = int(getattr(model.config, "n_routed_experts",
                            getattr(model.config, "num_experts", 0)))
        scale_provider = make_scale_provider(deploy_dir, n_exp)
        if rank == 0:
            print(f"  [deploy-scale] 专家 SSZ scale 对齐: {deploy_dir} (E={n_exp})", flush=True)
    if args.skip_wrap:
        # 隔离试验②（§6.7 C4-32 SIGSEGV）：无 STE/LoRA/快照，仅末层 attention 2D 可训练，
        # 保留 ZeRO-3 完整 engine+step 与 4 节点 HCCL 专家权重 gather 压力
        for p in model.parameters():
            p.requires_grad = False
        last = model.model.layers[-1]
        for n, p in last.named_parameters():
            if "self_attn" in n:
                p.requires_grad = True
        train_params = [p for p in model.parameters() if p.requires_grad]
        wrapped = []
        if rank == 0:
            n_train = sum(p.numel() for p in train_params)
            print(f"  [skip-wrap] 隔离模式：末层 attention 2D 可训练 {n_train/1e6:.1f}M（无 STE/LoRA）",
                  flush=True)
    else:
        wrapped = wrap_all_experts_glm(model, rank=args.rank, dtype=torch.bfloat16,
                                       zero3_gather=True, scale_provider=scale_provider)
        for i, w in enumerate(wrapped):
            w._qad_idx = i
    scale_only = os.environ.get("QAD_SCALE_ONLY") == "1"
    for p in model.parameters():
        p.requires_grad = False
    for w in wrapped:
        for p in w.lora.params.values():
            p.requires_grad = not scale_only   # scale-only 对照臂（§5.14-B1）：LoRA 冻结
        for p in getattr(w, "scale_params", {}).values():
            p.requires_grad = True             # 可训练 scale（QAD_TRAIN_SCALES=1 时注册）
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    if rank == 0:
        mode = "scale-only" if scale_only else "LoRA(+scale if enabled)"
        print(f"  GLM MoE 层 STE 包装 {len(wrapped)} 个（dense 跳过）；可训练 {n_train/1e6:.1f}M（{mode}）",
              flush=True)

    def _make_ckpt_fwd(orig_fwd):
        if diag_ep.ENABLED:
            def _ckpt_fwd(*a, **kw):
                _t0 = time.time()
                _out = torch.utils.checkpoint.checkpoint(
                    orig_fwd, *a, use_reentrant=False, **kw)
                diag_ep.note_layer_fwd(time.time() - _t0)
                return _out

            return _ckpt_fwd

        def _ckpt_fwd(*a, **kw):
            return torch.utils.checkpoint.checkpoint(
                orig_fwd, *a, use_reentrant=False, **kw)
        return _ckpt_fwd

    for layer in model.model.layers:
        layer.forward = _make_ckpt_fwd(layer.forward)

    if args.opt_mode == "sgd_stub":
        # C4-24 探针专用：最小优化器，避开 AdamW fp32 m+v 的 OOM 推力
        opt = torch.optim.SGD(train_params, lr=max(args.lr, 1e-6))
    elif args.opt_mode == "adamw_bf16":
        # 生产口径近似：AdamW（m+v 依 deepspeed zero3 的 dtype 选择 bf16 ~= fp16 语义）
        opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    else:
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
    if os.environ.get("C4_PROBE"):
        print(f"  [c4] engine ready: HBM alloc={torch.npu.memory_allocated()/2**30:.2f}GiB "
              f"max={torch.npu.max_memory_allocated()/2**30:.2f}GiB | "
              f"host RSS peak={_read_peak_rss_gib():.1f}GiB", flush=True)
    if os.environ.get("STRIP_DS_HOOKS") == "1":
        # 方案 B（§3.5-C1）：STE 经方案 A2 CPU 快照自管专家参数，DS 的 forward 期
        # all_gather 纯浪费（Qwen3 实测 58GB/forward，GLM ~7×）——剥离之；dense/attention 保留。
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

    # S2 路径（§6e-5）：不再做自管 ds_tensor 快照——_effective_all 直接读 LIVE
    # engine-managed ds_tensor（调研文档 §10.5 已证 forward 期数值正确）。
    # snapshot_ds_tensors() 保留为 no-op，此处不调用。
    if rank == 0:
        print(f"  [S2] 专家权重走 engine-managed ds_tensor（无自管快照）", flush=True)

    lora_state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            lora_state[n] = p
    if args.save_every:
        os.makedirs(args.save_dir, exist_ok=True)

    def save_ckpt(step_done: int, tag: str):
        # 仅 rank0 落盘：共享盘上 32 rank 并发写同一文件 = 损坏/异常风险（09-19）；
        # tmp+rename 原子写（同 t7 模式，防半写）；barrier 对齐全体 rank 时序
        def _shard(p):
            dt = getattr(p, 'ds_tensor', None)
            if dt is None:
                return p.detach().to(torch.bfloat16).contiguous().view(-1)
            if p.ds_status == ZeroParamStatus.AVAILABLE and p.data.numel() == p.ds_numel:
                flat = p.detach().to(torch.bfloat16).contiguous().view(-1)
                lo = min(dist.get_rank() * dt.numel(), max(p.ds_numel - dt.numel(), 0))
                return flat.narrow(0, lo, dt.numel()).contiguous()
            return dt.detach().to(dev).contiguous().view(-1)

        state = {}
        for n, p in lora_state.items():
            dt = getattr(p, 'ds_tensor', None)
            numel = getattr(p, 'ds_numel', p.numel())
            shape = list(getattr(p, 'ds_shape', p.shape))
            shard = _shard(p)
            if dt is not None and world > 1:
                parts = [torch.empty_like(shard) for _ in range(world)]
                dist.all_gather(parts, shard)
                full = torch.cat(parts, dim=0)[:numel]
                del parts
            else:
                full = shard[:numel]
            if rank == 0:
                state[n] = full.view(shape).to(torch.bfloat16).cpu()
            del shard, full
        os.makedirs(args.save_dir, exist_ok=True)
        path = f"{args.save_dir}/lora_step{step_done}_{tag}.pt"
        tmp = path + ".tmp"
        if rank == 0:
            torch.save(state, tmp)
            os.replace(tmp, path)
        print(f"  saved {path}", flush=True)
        dist.barrier()

    losses = []
    t_begin = time.time()
    total_tok = 0
    n_exec = 0
    start_step = 0
    # 0927 修复 6（flush 模式旋钮）：none|end|dual（默认 none）。
    # 依据（0926-27 六轮对照）：pre-bwd flush 与 onset 强相关（有 dual：run5/6/7 全部
    # ~100 起爆；无：run3/4 ≥161-294）——pre-bwd flush 清空 allocator 迫使 bwd 的
    # ~1.2 万次分配/步全部走驱动新鲜映射，驱动页表碎片化加速（= M2 载体）。
    # OFFLOAD=0 下 cached-block 全程复用是最优（驱动 churn 最小化）。
    _flush_mode = os.environ.get("QAD_FLUSH_MODE", "none").lower()
    first[0].shape[0]   # 维度自检（防 silent shape 错配）
    diag_ep.install(print)
    _diag_lf_fwd = 0.0
    _prof = None
    for step in range(start_step, args.steps):
        if _TP_ON and rank == 0 and step == _TP_START:
            _prof = _tp_make()
            _prof.__enter__()
            print(f"[tprof] profiling steps {_TP_START}..{_TP_START + _TP_ACTIVE - 1}"
                  f" -> {_TP_OUT}（QAD_DIAG/QAD_PROFILE 应关闭）", flush=True)
        diag_ep.reset()
        t_fetch = time.time()
        inp_np, ids_np, lps_np = next(gen)
        n_exec = step + 1 - start_step
        ft_ms = (time.time() - t_fetch) * 1000
        inp = torch.from_numpy(inp_np).unsqueeze(0).to(dev)
        t_fwd = time.time()
        out = engine(input_ids=inp, use_cache=False)
        if _QAD_PROFILE:
            torch.npu.synchronize()
        dt_fwd = time.time() - t_fwd
        _diag_lf_fwd = diag_ep.snapshot_layer_fwd()
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
        if _flush_mode == "dual":
            # 修复 6 之 pre-bwd flush（仅 dual 模式；实测有害于 onset，见上注）
            torch.npu.empty_cache()
        t_bwd = time.time()
        engine.backward(loss)
        if _QAD_PROFILE:
            torch.npu.synchronize()
        dt_bwd = time.time() - t_bwd
        t_step = time.time()
        engine.step()
        if _QAD_PROFILE:
            torch.npu.synchronize()
        dt_step = time.time() - t_step
        if os.environ.get("C4_PROBE"):
            print(f"  [c4] step {step}: HBM max={torch.npu.max_memory_allocated()/2**30:.2f}GiB "
                  f"host RSS peak={_read_peak_rss_gib():.1f}GiB ({time.time()-t_begin:.0f}s cum)",
                  flush=True)
        loss_v = float(loss.item())
        losses.append(loss_v)
        total_tok += ids_np.shape[0]
        if step % 2 == 0 or step == args.steps - 1:
            gn = engine.get_global_grad_norm()
            gn = f"{float(gn):.3f}" if gn is not None else "n/a"
            if rank == 0:
                print(f"  step {step:3d}: loss={loss_v:.4f} grad_norm={gn} "
                      f"fetch={ft_ms:.0f}ms", flush=True)
        if _QAD_PROFILE and rank == 0:
            phases = {"fetch": ft_ms / 1e3, "fwd": dt_fwd, "bwd": dt_bwd, "step": dt_step}
            print(prof_ste.flush_step(rank, step, phases), flush=True)
            print(f"  [prof mem] HBM_alloc={torch.npu.memory_allocated()/2**30:.2f}GiB "
                  f"HBM_max={torch.npu.max_memory_allocated()/2**30:.2f}GiB "
                  f"host_RSS_peak={_read_peak_rss_gib():.1f}GiB", flush=True)
        if diag_ep.ENABLED and rank == 0:
            print(diag_ep.step_report(step, {
                "fetch": ft_ms / 1e3, "fwd": dt_fwd, "bwd": dt_bwd, "step": dt_step,
                "layer_fwd_at_fwd_end": _diag_lf_fwd}), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(step + 1, "p")
        del out, lg
        # 0927 修复 6：步末 flush 仅在 end/dual 模式（默认 none——OFFLOAD=0 下
        # cached-block 全程复用最优，见 start_step 处注）。
        if _flush_mode in ("end", "dual"):
            torch.npu.empty_cache()
        if _prof is not None and step == _TP_START + _TP_ACTIVE - 1:
            _prof.__exit__(None, None, None)
            _tp_report(_prof)
            _prof = None

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
