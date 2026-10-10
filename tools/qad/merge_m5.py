#!/usr/bin/env python3
"""M5-1: merge M4 per-expert LoRA deltas into the Qwen3-30B BF16 base, save merged weights.

Student semantics (§3.1): out = fakequant(W)·x + (B·A)·x (output-space, experts only, fused
gate_up rows = gate|up, matching ste_experts.py chunk(2)). M5-1 merges W' = W + B·A so the
deployment requantization (GPTQ W4A8 + dense W8A8) runs on the compensated weights (§3.5-B2).

Usage (glm5_qad_env):
  python3 merge_m5.py \
    --ckpt /home/yult/llm_infer_workspace/ylt/glm52-qad/ckpt_m4/step_012000.pt \
    --base /home/yult/llm_infer_workspace/lichc/models/Qwen3-30B-A3B-Instruct-2507 \
    --out  /home/yult/llm_infer_workspace/ylt/glm52-qad/merged_m5
"""

import argparse
import os
import time

import torch


def get_weight(module, name):
    """Return the weight tensor for a fused Parameter or an nn.Linear."""
    p = getattr(module, name)
    return p if isinstance(p, torch.Tensor) else p.weight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[merge] loading ckpt {args.ckpt} ...", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["sd"]
    print(f"[merge] ckpt step={ck.get('step')} lora tensors={len(sd)}", flush=True)

    print(f"[merge] loading base BF16 model (cpu) {args.base} ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[merge] base loaded ({time.time()-t0:.0f}s)", flush=True)

    n_layer = 0
    n_expert = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not name.endswith(".mlp.experts"):
                continue
            n_layer += 1
            pre = name + ".lora.params."
            Agu = sd[pre + "Agu"].float()  # [E, r, hidden]
            Bgu = sd[pre + "Bgu"].float()  # [E, 2*mi, r]
            Adn = sd[pre + "Adn"].float()  # [E, r, mi]
            Bdn = sd[pre + "Bdn"].float()  # [E, hidden, r]
            E, r, hidden = Agu.shape
            two_mi = Bgu.shape[1]
            mi = two_mi // 2

            fused = hasattr(module, "gate_up_proj")
            if fused:
                gu = get_weight(module, "gate_up_proj")
                assert tuple(gu.shape) == (E, two_mi, hidden), (name, tuple(gu.shape))
                dn = get_weight(module, "down_proj")
                assert tuple(dn.shape) == (E, hidden, mi), (name, tuple(dn.shape))
                for e in range(E):
                    gu[e] += (Bgu[e] @ Agu[e]).to(gu.dtype)
                    dn[e] += (Bdn[e] @ Adn[e]).to(dn.dtype)
            else:
                gp = get_weight(module, "gate_proj")  # [E, mi, hidden]
                up = get_weight(module, "up_proj")
                dn = get_weight(module, "down_proj")
                assert tuple(gp.shape) == (E, mi, hidden), (name, tuple(gp.shape))
                for e in range(E):
                    dgu = (Bgu[e] @ Agu[e]).to(gp.dtype)  # [2mi, hidden]
                    gp[e] += dgu[:mi]
                    up[e] += dgu[mi:]
                    dn[e] += (Bdn[e] @ Adn[e]).to(dn.dtype)
            n_expert += E
    print(f"[merge] merged {n_layer} layers, {n_expert} expert matrices "
          f"(fused={fused}, {time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    print(f"[merge] saving merged model to {args.out} ...", flush=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print(f"[merge] DONE ({time.time()-t0:.0f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
