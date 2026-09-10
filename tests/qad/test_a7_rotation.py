"""A7 消融：Hadamard 旋转量化期全模拟（调研文档 §6.1）。

目标：拿到"旋转在 AWQ 前端叠加后的精度上限"（tiny 端到端 wikitext ppl +
专家 norm_loss），回答"是否值得投入 CANN kernel"。**非目标**：可部署权重
（产物不可部署，部署需在线 kernel，§2.5）。

方案要点（与 §6.1 Step 0-6 对应，tiny 适配差异见各函数 docstring）：
- 量化期全模拟：权重侧折叠 + hook 注入（apply_rotation），fp 前向等价性闸门 <1e-3
- R1 注入在 RMSNorm 之后（gain g 不满足旋转不变性，norm 前旋转破坏等价）
- mini 量化器复刻主线三段式语义的 RTN 简化版：统计采集（x_abs_mean，基于
  折叠后权重产生的激活）→ AWQ α 网格搜索（MSE(RTN(w·s)/s, w) 择优）→
  int4 per-channel RTN（无 GPTQ Hessian 补偿——mini 简化，两组公平对照）
- 对照组 AWQ-only 同批次同数据同 seed，唯一变量是旋转
- 消融矩阵：R1-only / R2-only / R1+R2 × R1 block {full, 128}
  （tiny inter=128 使 R2 的 block 256 档不可行，协议尺寸限制非算法限制）

运行（npuslim conda env，CPU）：
    /data/yult/miniconda3/envs/npuslim/bin/python tests/qad/test_a7_rotation.py
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from npuslim.qad.rotation import QuaRotMode, create_rot
from npuslim.qad.tiny_moe import TinyMoEConfig, TinyMoEModel, apply_rotation

WIKITEXT_DIR = Path(__file__).resolve().parents[2] / "workspace" / "hf_home_4task" / "hub" / \
    "datasets--EleutherAI--wikitext_document_level" / "snapshots" / \
    "647234772b9554e208af6c826f23b99e3cac88c8" / "wikitext-2-raw-v1"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------- 数据：wikitext-2 byte-level ----------------

def load_stream(split: str) -> torch.Tensor:
    """wikitext-2 parquet → utf-8 byte 流（vocab 256，无 tokenizer 依赖）。

    本地缓存是 EleutherAI document-level 变体，文本列名为 'page'。
    """
    df = pd.read_parquet(WIKITEXT_DIR / f"wikitext-2-raw-v1-{split}.parquet")
    text = "\n\n".join(df["page"].tolist())[:2_000_000]  # 截断控制 CPU 训练时长
    data = torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).long()
    print(f"       wikitext-2/{split}: {len(data)} bytes")
    return data


def sample_windows(stream: torch.Tensor, n: int, seq: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(stream) - seq - 1, (n,), generator=g)
    return torch.stack([stream[o:o + seq] for o in offs])


# ---------------- mini 量化器（主线三段式的 RTN 简化） ----------------

def rtn_int4(w: torch.Tensor) -> torch.Tensor:
    """int4 per-channel 对称 RTN（复刻部署语义：q ∈ [-8,7]，无 per-group）。"""
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
    return (w / s).round().clamp(-8, 7) * s


def awq_quantize(w: torch.Tensor, x_mean: torch.Tensor, alphas: int = 20
                 ) -> tuple[torch.Tensor, float, float]:
    """AWQ α 网格搜索 + 折叠（§1.1：s = x_mean^α 几何归一化，MSE(RTN(w·s)/s, w) 择优）。

    返回 (量化后权重, best_α, norm_loss)。α=0 时 s≡1，退化为纯 RTN。
    """
    best_s, best_loss, best_a = None, float("inf"), 0.0
    for a in torch.linspace(0.0, 1.0, alphas).tolist():
        s = (x_mean + 1e-6) ** a
        s = s / s.mean()
        eff = rtn_int4(w * s.unsqueeze(0)) / s.unsqueeze(0)
        loss = (eff - w).pow(2).sum().item()
        if loss < best_loss:
            best_s, best_loss, best_a = s, loss, a
    w_q = rtn_int4(w * best_s.unsqueeze(0)) / best_s.unsqueeze(0)
    n_loss = (w_q - w).norm().item() / w.norm().clamp_min(1e-8).item()
    return w_q, best_a, n_loss


def inject_outliers(model: TinyMoEModel, n_channels: int = 8, factor: float = 30.0,
                    seed: int = 5) -> torch.Tensor:
    """合成激活离群（tiny 适配，QuaRot/AWQ 论文同款动机）。

    真实 LLM 训练后期在残差流少数通道上出现大幅离群激活，是旋转量化要
    解决的核心问题；tiny byte-level 模型训不出该结构（实测 per-channel
    abs-mean 近平坦，AWQ α 搜索退化为 0，消融无分辨力）。故在训练前对
    embed 的少数通道放大——残差流通道 k 在所有 token 上离群，模型在含
    离群分布上学习出与真实 LLM 同构的"离群 + 补偿"权重结构。
    """
    g = torch.Generator().manual_seed(seed)
    chans = torch.randperm(model.cfg.hidden_size, generator=g)[:n_channels]
    with torch.no_grad():
        model.embed.weight.data[:, chans] *= factor
    return chans


def outlier_ratio(x: torch.Tensor) -> float:
    """激活离群度：per-channel abs-mean 的 max/mean（>>1 说明存在离群通道）。"""
    cm = x.reshape(-1, x.shape[-1]).abs().mean(dim=0)
    return (cm.max() / cm.mean()).item()


def collect_stats(model: TinyMoEModel, calib: torch.Tensor) -> list[dict]:
    """统计采集：专家 x_abs_mean（gate_up 输入与 down 输入，旋转后基）。

    与 §1.1 一致：统计必须基于折叠后权重产生的激活——本方案 hook 在前向中
    即时旋转激活，采集到的就是旋转后统计，无需二次变换。
    """
    accum = [{"gu_sum": None, "gu_n": 0, "dn_sum": None, "dn_n": 0}
             for _ in range(sum(1 for _ in model.iter_experts()))]
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(calib[i:i + 1])
            for e in model.iter_experts():
                a = accum[e.expert_id]
                for key, t in (("gu", e._last_in), ("dn", e._last_h)):
                    v = t.abs().sum(dim=0)
                    if a[f"{key}_sum"] is None:
                        a[f"{key}_sum"] = v.clone()
                    else:
                        a[f"{key}_sum"] += v
                    a[f"{key}_n"] += t.shape[0]
    stats = []
    for a in accum:
        stats.append({
            "gate_up": a["gu_sum"] / max(a["gu_n"], 1),
            "down": a["dn_sum"] / max(a["dn_n"], 1),
        })
    return stats


def quantize_arm(model: TinyMoEModel, calib: torch.Tensor) -> dict:
    """单组量化：统计 → AWQ 搜索 → 折叠写回 → 开启激活量化模拟（就地量化）。

    AWQ α 搜索保持主线语义的轻量目标 MSE(RTN(w·s)/s, w)（权重侧）；
    激活 per-token int8 fake-quant 对所有 arm 统一开启（W4A8 语义），
    各组唯一变量是 AWQ α 与旋转。激活量化是旋转收益的主战场：离群通道
    拉大 per-token scale、压低其余通道精度，旋转白化后该误差源消失。
    """
    stats = collect_stats(model, calib)
    metrics = {"gate_up": [], "down": [], "alpha_gate_up": [], "alpha_down": []}
    with torch.no_grad():
        for e, st in zip(model.iter_experts(), stats):
            for proj in ("gate_up", "down"):
                w = getattr(e, proj).data
                w_q, alpha, n_loss = awq_quantize(w, st[proj])
                getattr(e, proj).data = w_q
                metrics[proj].append(n_loss)
                metrics[f"alpha_{proj}"].append(alpha)
    metrics["norm_loss"] = (sum(metrics["gate_up"]) / len(metrics["gate_up"]),
                            sum(metrics["down"]) / len(metrics["down"]))
    metrics["alpha_mean"] = (sum(metrics["alpha_gate_up"]) / len(metrics["alpha_gate_up"]),
                             sum(metrics["alpha_down"]) / len(metrics["alpha_down"]))
    gu0 = stats[0]["gate_up"]
    metrics["outlier"] = (gu0.max() / gu0.mean()).item()  # AWQ 统计的通道离群度
    for e in model.iter_experts():
        e.activation_quant = True
        e.act_bits = 4  # 压力测试放大镜：int8 误差低于 tiny ppl 分辨力（见 fake_quant_act_per_token）
    return metrics


def eval_ppl(model: TinyMoEModel, stream: torch.Tensor, n: int = 16, seq: int = 256) -> float:
    windows = sample_windows(stream, n, seq, seed=123)
    nll_sum, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            tok = windows[i:i + 1]
            logits = model(tok)[0, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits, tok[0, 1:], reduction="sum")
            nll_sum += nll.item()
            n_tok += seq - 1
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


# ---------------- 旋转配置 ----------------

def make_rotation(cfg: TinyMoEConfig, use_r1: bool, use_r2: bool,
                  r1_block: int, seed: int = 2026):
    r1 = create_rot(QuaRotMode.HADAMARD, cfg.hidden_size,
                    block_size=r1_block if r1_block else -1, seed=seed) if use_r1 else None
    r2s = [create_rot(QuaRotMode.HADAMARD, cfg.inter_size, seed=seed + 1000 + i)
           if use_r2 else None for i in range(cfg.num_layers)]
    return r1, r2s


# ---------------- 主流程 ----------------

def main() -> None:
    torch.manual_seed(0)
    cfg = TinyMoEConfig(vocab_size=256, hidden_size=256, inter_size=128,
                        num_experts=8, num_layers=6, num_heads=4)
    print(f"A7 tiny 模型: {cfg}  参数量: {sum(p.numel() for p in TinyMoEModel(cfg).parameters()):,}")

    # ---- Step 1-3 复检：seed 可复现 + 等价性闸门（A7 尺度）----
    r1, r2s = make_rotation(cfg, True, True, r1_block=None)
    check("a7.seed_reproducible",
          torch.equal(r1, create_rot(QuaRotMode.HADAMARD, cfg.hidden_size, seed=2026)))
    probe = TinyMoEModel(cfg).eval()
    tok = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        base = probe(tok)
    rot = copy.deepcopy(probe)
    apply_rotation(rot, r1, r2s)
    with torch.no_grad():
        gate = (rot(tok) - base).abs().max().item()
    check("a7.equivalence_gate", gate < 1e-3, f"max_diff={gate:.2e}")
    del probe, rot

    # ---- fp 基座：wikitext byte-level 训练 ----
    train_stream = load_stream("train")
    val_stream = load_stream("validation")
    fp = TinyMoEModel(cfg)
    chans = inject_outliers(fp)
    print(f"       离群注入: embed 通道 {chans.tolist()} ×30（模拟真实 LLM 激活离群结构）")
    opt = torch.optim.AdamW(fp.parameters(), lr=1e-3)
    seq, batch, steps = 128, 4, 250
    t0 = time.time()
    first = last = None
    for step in range(steps):
        offs = torch.randint(0, len(train_stream) - seq - 1, (batch,))
        tok = torch.stack([train_stream[o:o + seq] for o in offs])
        logits = fp(tok)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                                                 tok[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
        if (step + 1) % 50 == 0:
            print(f"       train step {step + 1}/{steps}  CE={last:.4f}  {time.time()-t0:.0f}s")
    check("a7.fp_train_decreases", last < first, f"CE {first:.4f} -> {last:.4f}")
    ppl_fp = eval_ppl(fp, val_stream)
    print(f"       fp 基座 val ppl = {ppl_fp:.2f}")

    calib = sample_windows(train_stream, 4, 256, seed=777)  # 各组共用同一校准数据

    # ---- Step 4-6：消融矩阵 ----
    arms = [
        ("rtn_only",        False, False, None),
        ("awq_only",        False, False, None),
        ("r1_full",         True,  False, None),
        ("r1_block128",     True,  False, 128),
        ("r2",              False, True,  None),
        ("r1_full+r2",      True,  True,  None),
        ("r1_block128+r2",  True,  True,  128),
    ]
    results = {}
    for name, use_r1, use_r2, r1_block in arms:
        m = copy.deepcopy(fp)
        r1, r2s = make_rotation(cfg, use_r1, use_r2, r1_block)
        if r1 is not None or r2s[0] is not None:
            apply_rotation(m, r1, r2s)
        metrics = quantize_arm(m, calib)
        ppl = eval_ppl(m, val_stream)
        results[name] = {"ppl": ppl, "norm_loss": metrics["norm_loss"],
                         "alpha": metrics["alpha_mean"]}
        gu, dn = metrics["norm_loss"]
        a_gu, a_dn = metrics["alpha_mean"]
        print(f"  [{name:>16}] ppl={ppl:8.2f}  norm_loss(gu/dn)={gu:.4f}/{dn:.4f}  "
              f"alpha(gu/dn)={a_gu:.2f}/{a_dn:.2f}  stat_outlier={metrics['outlier']:.1f}")

    # ---- 汇总与判据 ----
    base_ppl = results["awq_only"]["ppl"]
    print("-" * 76)
    print(f"fp 基座 ppl={ppl_fp:.2f} | awq_only 基线 ppl={base_ppl:.2f}")
    verdict, best = "未达立项阈值（>2pp 相对改善）", None
    for name, r in results.items():
        if name in ("rtn_only", "awq_only"):
            continue
        gain = (base_ppl - r["ppl"]) / base_ppl * 100
        print(f"  {name:>16}: ppl 改善 {gain:+.2f}%")
        if best is None or r["ppl"] < best[1]:
            best = (name, r["ppl"])
    if best and (base_ppl - best[1]) / base_ppl > 0.02:
        verdict = f"达立项阈值：{best[0]} 改善 {(base_ppl-best[1])/base_ppl*100:.1f}% → 建议进入 §2.5 kernel 评估"
    print(f"A7 结论（tiny 规模，外推需谨慎）: {verdict}")
    check("a7.matrix_completed", len(results) == len(arms) and all(v["ppl"] > 0 for v in results.values()))


if __name__ == "__main__":
    main()
    print("-" * 76)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("A7 ABLATION RUN COMPLETED")
