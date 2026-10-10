
"""A7 Step B 评测：真实规模 W4A8 语义 ppl 对照（ROT vs NOROT，调研文档 §6.1 Step 5）。

流程（自建路径，部署无关）：
1. 加载 BF16 源模型（Qwen3MoeForCausalLM）
2. dequant 量化产物（打包 nibble 补码 / W8A8 int8）→ float 权重，重建模型权重
   - W4A8 专家（per-expert 2D gate/up/down，nibble 打包）：
     q = 补码(nibble)（先转 int32 防 uint8 回绕）；w[2i]=(q_low)*scale[2i]；w[2i+1]=(q_high)*scale[2i+1]
     已用无 AWQ tiny 运行对源权重验证 rel_err 0.14-0.21（4-bit GPTQ 合理范围）
   - W8A8 稠密（q/k/v/o_proj）：w = q * scale（rel_err 0.011）
   - FLOAT（norm/embed/router/lm_head）：直接取产物 bf16 张量
   - scale_bias/weight_offset 不进入 float 前向（kernel 激活表征补偿，§10.9）
3. eval hook：
   - ROT 臂：input_layernorm / post_attention_layernorm 输出 × R1.T（与量化期一致）
   - 两臂共同：上述 norm 输出 per-token int8 动态 fake-quant（W4A8 部署语义）
     （旋转收益主战场 = 量化边界激活离群，Step A：attn 48× / mlp 26×）
4. wikitext-2 窗口前向 → CE → ppl

判定：ROT ppl 相对 NOROT 改善是否显著（对照 AWQ-only 基线，§6.1 判据）。

运行（npuslim env）：
    python -m npuslim.qad.eval_rotated_ppl --checkpoint <stepB/rot> --model-path <bf16> \
        --rotated --device npu --output stepB_rot_ppl.json
"""
from __future__ import annotations
# ===== Phase D 路径参数化（env 带默认回退）=====
import os as _pd_os
SHARED = _pd_os.environ.get("SHARED", "/llm_infer_workspace")
SHARED_HOST = _pd_os.environ.get("SHARED_HOST", "/home/yult/llm_infer_workspace")
YLT = _pd_os.environ.get("YLT", SHARED + "/ylt")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from npuslim.qad.rotation import QuaRotMode, create_rot
from npuslim.qad.rotation_quant import register_rotation_hooks


# ---------------- dequant ----------------

def _two_complement(nib: torch.Tensor) -> torch.Tensor:
    """4-bit 补码解码（先转 int32，防 uint8 回绕：15-16 必须=-1 而非 255）。"""
    n = nib.to(torch.int32)
    return torch.where(n < 8, n, n - 16).float()


def dequant_checkpoint(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """将量化产物反量化为 float 权重 dict：{checkpoint_key(去.weight): float32 tensor}。"""
    ckpt = Path(checkpoint_dir)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    wm: Dict[str, str] = index["weight_map"]
    shard_cache: Dict[str, dict] = {}

    def load(name: str) -> torch.Tensor:
        shard = wm[name]
        if shard not in shard_cache:
            from safetensors.torch import load_file
            shard_cache[shard] = load_file(str(ckpt / shard))
        return shard_cache[shard][name]

    out: Dict[str, torch.Tensor] = {}
    # 先收集所有带 weight_scale 的量化模块
    quant_mods: Dict[str, List[str]] = {}  # mod_key -> [weight, weight_scale, scale_bias?]
    for name in wm:
        if name.endswith(".weight") and (name[:-len(".weight")] + ".weight_scale") in wm:
            mod = name[:-len(".weight")]
            quant_mods[mod] = [name]
            for suf in (".weight_scale", ".scale_bias", ".weight_offset"):
                if mod + suf in wm:
                    quant_mods[mod].append(mod + suf)

    for mod, names in quant_mods.items():
        packed = load(mod + ".weight")
        scale = load(mod + ".weight_scale").float()
        if packed.shape[0] == scale.shape[0] // 2:
            # W4A8 打包 nibble [out//2, in]
            p = packed.to(torch.uint8)
            out_dim = scale.shape[0]
            w = torch.empty(out_dim, p.shape[1], dtype=torch.float32)
            w[0::2] = _two_complement(p & 0xF) * scale[0::2]
            w[1::2] = _two_complement(p >> 4) * scale[1::2]
        else:
            # W8A8 int8 [out, in]
            w = packed.float() * scale
        out[mod] = w

    # FLOAT 张量（无 weight_scale）：norm / embed / router / lm_head / final norm
    for name in wm:
        if name.endswith(".weight") and name[:-len(".weight")] not in out and \
           name[:-len(".weight")] + ".weight_scale" not in wm:
            out[name[:-len(".weight")]] = load(name).float()
    return out


# ---------------- 权重重建 ----------------

def rebuild_model_weights(model, dequant: Dict[str, torch.Tensor], device: Optional[torch.device] = None) -> None:
    """把 dequant 权重装进 HF 模型（专家 per-expert 2D -> 3D gate_up/down 重建）。

    device：目标设备（模型参数所在设备）。dequant 为 CPU fp32，
    赋 .data 前必须显式搬运，否则 NPU/CUDA 评测时前向设备不匹配。
    """
    def put(mod: str, ref: torch.Tensor) -> None:
        t = dequant[mod].to(device=ref.device if device is None else device, dtype=ref.dtype)
        ref.data = t

    layers = model.model.layers
    for i, layer in enumerate(layers):
        prefix = f"model.layers.{i}"
        # attention 2D
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = f"{prefix}.self_attn.{proj}"
            if mod in dequant:
                put(mod, getattr(layer.self_attn, proj).weight)
        # router gate（FLOAT，来自产物）
        mod = f"{prefix}.mlp.gate"
        if mod in dequant:
            put(mod, layer.mlp.gate.weight)
        # experts：per-expert 2D -> 3D gate_up_proj / down_proj
        experts = layer.mlp.experts
        n_exp = experts.num_experts if hasattr(experts, "num_experts") else \
            experts.gate_up_proj.shape[0]
        if n_exp and hasattr(experts, "gate_up_proj"):
            gu = torch.stack([
                torch.cat([dequant[f"{prefix}.mlp.experts.{e}.gate_proj"],
                           dequant[f"{prefix}.mlp.experts.{e}.up_proj"]], dim=0)
                for e in range(n_exp)
            ])
            ref = experts.gate_up_proj
            ref.data = gu.to(device=ref.device if device is None else device, dtype=ref.dtype)
        if n_exp and hasattr(experts, "down_proj"):
            dn = torch.stack([dequant[f"{prefix}.mlp.experts.{e}.down_proj"]
                              for e in range(n_exp)])
            ref = experts.down_proj
            ref.data = dn.to(device=ref.device if device is None else device, dtype=ref.dtype)
    # pre/post 模块
    if "model.embed_tokens" in dequant:
        put("model.embed_tokens", model.model.embed_tokens.weight)
    if "model.norm" in dequant:
        put("model.norm", model.model.norm.weight)
    if "lm_head" in dequant and hasattr(model, "lm_head"):
        put("lm_head", model.lm_head.weight)


# ---------------- eval hooks ----------------

def fake_quant_act_per_token(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """per-token 动态 int8 fake-quant（别名 → npuslim.qad.ste.activation_dynamic_int8，单一语义源）。"""
    from npuslim.qad.ste import activation_dynamic_int8
    return activation_dynamic_int8(x, bits)


def register_eval_hooks(model, r1: Optional[torch.Tensor], act_bits: int) -> List:
    """注册评测 hook：R1 旋转（ROT 臂）+ per-token 激活 fake-quant（两臂）。

    挂在 norm 输出（量化边界）：input_layernorm -> q/k/v；post_attention_layernorm -> router/gate_up。
    返回 handle 列表（调用方负责 remove）。
    """
    handles = []
    r1_t = r1.T.float() if r1 is not None else None

    def make_hook(rotate: bool):
        def hook(module, inp, out):
            x = out
            if rotate:
                x = (x.float() @ r1_t).to(x.dtype)
            return fake_quant_act_per_token(x, act_bits)
        return hook

    for layer in model.model.layers:
        handles.append(layer.input_layernorm.register_forward_hook(make_hook(r1 is not None)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(make_hook(r1 is not None)))
    return handles


# ---------------- wikitext ppl ----------------

def load_wikitext(split: str = "validation", max_chars: int = 2_000_000,
                  wikitext_dir: Optional[str] = None) -> str:
    import pandas as pd
    if wikitext_dir:
        base = Path(wikitext_dir)
    else:
        # 数据源：WIKITEXT_HUB env 优先；默认回退 5090 本机 hf cache（历史调试路径，
        # C3 上运行需设 WIKITEXT_HUB 或显式传 wikitext_dir）
        base = Path(_pd_os.environ.get(
            "WIKITEXT_HUB",
            "/data/yult/llm_inference/npuslim/workspace/hf_home_4task/hub/"
            "datasets--EleutherAI--wikitext_document_level/snapshots/"
            "647234772b9554e208af6c826f23b99e3cac88c8/wikitext-2-raw-v1"))
    df = pd.read_parquet(base / f"wikitext-2-raw-v1-{split}.parquet")
    return "\n\n".join(df["page"].tolist())[:max_chars]


def eval_ppl(model, tokenizer, n_windows: int = 16, seq: int = 256, seed: int = 123,
             device: torch.device = None, wikitext_dir: Optional[str] = None) -> float:
    text = load_wikitext("validation", wikitext_dir=wikitext_dir)
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    ids = enc["input_ids"][0]
    g = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(ids) - seq - 1, (n_windows,), generator=g)
    windows = torch.stack([ids[o:o + seq] for o in offs]).to(device)
    nll_sum, n_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(n_windows):
            logits = model(windows[i:i + 1]).logits[0, :-1].float()
            nll = F.cross_entropy(logits, windows[i, 1:], reduction="sum")
            nll_sum += nll.item()
            n_tok += seq - 1
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A7 Step B: W4A8 语义 ppl 评测（ROT/NOROT 对照）")
    parser.add_argument("--checkpoint", default=None, help="量化产物目录（stepB/rot 或 norot）")
    parser.add_argument("--model-path", default=f"{SHARED}/lichc/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--model-hub", default="ms")
    parser.add_argument("--rotated", action="store_true", help="应用 R1 旋转 hook（ROT 臂）")
    parser.add_argument("--r1-seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=-1)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--skip-dequant", action="store_true",
                        help="D-a/D-0 模式：不加载产物，直接用 BF16 源权重（激活 fake-quant 仍默认生效）")
    parser.add_argument("--no-act-quant", action="store_true",
                        help="D-b/D-0 模式：不注册激活 fake-quant hook（fp 激活；同时意味着无旋转）")
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--wikitext-dir", default=None, help="wikitext-2-raw-v1 parquet 目录（节点路径）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.checkpoint is None and not args.skip_dequant:
        parser.error("必须提供 --checkpoint，或使用 --skip-dequant（BF16 源权重模式）")
    if args.no_act_quant and args.rotated:
        print("[eval] 警告: --no-act-quant 与 --rotated 同时设置——旋转 hook 位于激活路径，"
              "fp 激活模式下旋转不生效（按纯权重-only 口径执行）", flush=True)

    if args.device == "npu":
        device_map = {"": "npu:0"}
    elif args.device == "cuda":
        device_map = {"": "cuda:0"}
    else:
        device_map = args.device

    t0 = time.time()
    print(f"[eval] 加载 BF16 源模型 {args.model_path} (device_map={device_map}) ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device_map, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = model.model.embed_tokens.weight.device
    print(f"[eval] 模型加载完成 {time.time()-t0:.0f}s, device={device}", flush=True)

    if not args.skip_dequant:
        print(f"[eval] dequant 量化产物 {args.checkpoint} ...", flush=True)
        dequant = dequant_checkpoint(args.checkpoint)
        print(f"[eval] dequant 完成: {len(dequant)} 个模块", flush=True)
        rebuild_model_weights(model, dequant, device=device)
    else:
        print("[eval] --skip-dequant: 使用 BF16 源权重（D-a 激活-only / D-0 锚点）", flush=True)

    handles = []
    if args.rotated:
        hidden = model.config.hidden_size
        r1 = create_rot(QuaRotMode.HADAMARD, hidden, block_size=args.block_size,
                        seed=args.r1_seed).float()
        r1 = r1.to(device)
        print(f"[eval] ROT 臂: R1 {r1.shape} 注入 + 激活 W{args.act_bits} fake-quant", flush=True)
    else:
        r1 = None
        print(f"[eval] NOROT 臂: 激活 W{args.act_bits} fake-quant（无旋转）", flush=True)
    if args.no_act_quant:
        handles = []
        print("[eval] --no-act-quant: fp 激活，不注册任何 hook（D-b 权重-only / D-0 锚点）", flush=True)
    else:
        handles = register_eval_hooks(model, r1, args.act_bits)

    print(f"[eval] wikitext-2 ppl ({args.n_windows} x {args.seq}) ...", flush=True)
    ppl = eval_ppl(model, tokenizer, n_windows=args.n_windows, seq=args.seq, device=device,
                   wikitext_dir=args.wikitext_dir)
    for h in handles:
        h.remove()

    result = {
        "checkpoint": args.checkpoint, "rotated": args.rotated,
        "skip_dequant": args.skip_dequant, "no_act_quant": args.no_act_quant,
        "ppl": round(ppl, 4), "n_windows": args.n_windows, "seq": args.seq,
        "act_bits": args.act_bits, "device": str(device),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[eval] ppl = {ppl:.4f}  (用时 {time.time()-t0:.0f}s)")
    out_path = args.output or "stepB_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval] 结果已写入 {out_path}")


if __name__ == "__main__":
    main()
