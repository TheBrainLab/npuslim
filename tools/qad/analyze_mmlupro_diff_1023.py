#!/usr/bin/env python3
"""Targeted analysis of the BF16 vs w4a8c8 MMLU-Pro 500 discordant pairs."""
import argparse, json, math
from collections import Counter, defaultdict

def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)

def acc(correct, n):
    return correct / n if n else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--w4a8c8", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    B = {r["question_id"]: r for r in map(json.loads, open(args.bf16))}
    W = {r["question_id"]: r for r in map(json.loads, open(args.w4a8c8))}
    qids = sorted(set(B) & set(W))
    assert len(qids) == len(B) == len(W), (len(B), len(W), len(qids))

    cats = defaultdict(lambda: {"n": 0, "bf16_correct": 0, "w4_correct": 0,
                                "bf16_only": 0, "w4_only": 0})
    disc = []
    for q in qids:
        b, w = B[q], W[q]
        c = b["category"]
        cats[c]["n"] += 1
        cats[c]["bf16_correct"] += int(bool(b["correct"]))
        cats[c]["w4_correct"] += int(bool(w["correct"]))
        if bool(b["correct"]) != bool(w["correct"]):
            disc.append((q, c, b, w))
            if b["correct"]:
                cats[c]["bf16_only"] += 1
            else:
                cats[c]["w4_only"] += 1

    b_only = [x for x in disc if x[2]["correct"]]
    w_only = [x for x in disc if x[3]["correct"]]
    b_correct = sum(bool(r["correct"]) for r in B.values())
    w_correct = sum(bool(r["correct"]) for r in W.values())

    cat_rows = []
    for c, v in cats.items():
        cat_rows.append({
            "category": c, **v,
            "bf16_acc": acc(v["bf16_correct"], v["n"]),
            "w4_acc": acc(v["w4_correct"], v["n"]),
            "net": v["bf16_only"] - v["w4_only"],
        })
    cat_rows.sort(key=lambda x: (-x["net"], -x["bf16_only"], x["category"]))

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        if not n:
            return None
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    def tok(r):
        return (r.get("usage") or {}).get("completion_tokens") or 0

    summary = {
        "n": len(qids),
        "bf16_correct": b_correct,
        "bf16_acc": acc(b_correct, len(qids)),
        "w4a8c8_correct": w_correct,
        "w4a8c8_acc": acc(w_correct, len(qids)),
        "discordant": len(disc),
        "bf16_only": len(b_only),
        "w4_only": len(w_only),
        "net_bf16_minus_w4": len(b_only) - len(w_only),
        "delta_pp": 100.0 * (acc(b_correct, len(qids)) - acc(w_correct, len(qids))),
        "mcnemar_exact_p": exact_mcnemar(len(b_only), len(w_only)),
        "bf16_only_unparsed_w4": sum(not bool(x[3].get("parsed")) for x in b_only),
        "w4_only_unparsed_bf16": sum(not bool(x[2].get("parsed")) for x in w_only),
        "bf16_only_w4_median_completion_tokens": median([tok(x[3]) for x in b_only]),
        "w4_only_bf16_median_completion_tokens": median([tok(x[2]) for x in w_only]),
        "categories": cat_rows,
        "bf16_only_ids": [x[0] for x in b_only],
        "w4_only_ids": [x[0] for x in w_only],
    }
    json.dump(summary, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    lines = []
    lines.append("# MMLU-Pro 500：BF16 vs w4a8c8 净差异定向分析（1023）\n")
    lines.append("> 数据：`mmlupro_bf16.jsonl` / `mmlupro_w4a8c8.jsonl`；同题 500 题配对。\n")
    lines.append("## 总体\n")
    lines.append("| 指标 | BF16 | w4a8c8 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 正确 | {b_correct}/500 | {w_correct}/500 |")
    lines.append(f"| 准确率 | {100*summary['bf16_acc']:.1f}% | {100*summary['w4a8c8_acc']:.1f}% |")
    lines.append(f"\n- Discordant：**{len(disc)}**（BF16-only **{len(b_only)}**，w4-only **{len(w_only)}**），净差 **{summary['net_bf16_minus_w4']:+d} 题 / {summary['delta_pp']:+.2f}pp**。")
    lines.append(f"- McNemar exact two-sided **p = {summary['mcnemar_exact_p']:.4f}**。")
    lines.append(f"- 未解析：BF16-only 中 w4 未解析 {summary['bf16_only_unparsed_w4']} 题；w4-only 中 BF16 未解析 {summary['w4_only_unparsed_bf16']} 题。")
    lines.append(f"- 完成 tokens 中位数：BF16-only 的 w4 答案 {summary['bf16_only_w4_median_completion_tokens']}；w4-only 的 BF16 答案 {summary['w4_only_bf16_median_completion_tokens']}。\n")
    lines.append("## 分类净差\n")
    lines.append("| 类别 | n | BF16 正确 | w4 正确 | BF16-only | w4-only | 净差 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in cat_rows:
        lines.append(f"| {r['category']} | {r['n']} | {r['bf16_correct']} | {r['w4_correct']} | {r['bf16_only']} | {r['w4_only']} | {r['net']:+d} |")
    lines.append("\n## 判读\n")
    lines.append(f"- 净差集中在 **law（+6）** 与 **engineering（+5）**，两者合计 +11，占 18 题的 61%；chemistry 正负相抵（3 vs 3），physics 反而 -1。")
    lines.append("- 分类样本小（law n=46，engineering n=40），只能作为定位线索，不能单类下显著性结论；总体 McNemar 才是主判据。")
    lines.append("- 该缺口是 **知识轴的小缺口**（3.6pp），不改变 l0 主推荐；w4a8c8 的风险条款应写明“law/engineering 为主要贡献类”。")
    lines.append("- 统计口径：exact McNemar **p=0.0198**；此前文档的 p=0.014 是未做连续性校正的 χ²，方向与显著性结论不变，但以 exact 为准。")
    open(args.out_md, "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("n","bf16_acc","w4a8c8_acc","bf16_only","w4_only","delta_pp","mcnemar_exact_p")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
