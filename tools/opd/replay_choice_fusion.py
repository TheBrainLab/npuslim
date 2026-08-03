#!/usr/bin/env python3
"""Replay overlap-aware OPD expert fusion on choice-task score files.

Input JSONL format, one sample per line:

{
  "id": "optional-sample-id",
  "target": 0,
  "student_scores": [1.2, 0.4, -0.3, 0.1],
  "teacher_scores": [1.0, 0.7, -0.4, 0.0],
  "experts": {
    "multi_choice": [1.1, 0.5, -0.2, 0.2],
    "task": [1.3, 0.2, -0.5, 0.0]
  }
}

The script implements:

    z_final = z_student + sum_i lambda_i * (z_expert_i - z_student)

where lambda_i is computed by npuslim.opd.OverlapAwareRouter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from npuslim.opd import ExpertScoreDelta, OverlapAwareRouter, fuse_candidate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Candidate score JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Path for per-sample replay details.")
    parser.add_argument("--summary-json", required=True, help="Path for aggregate metrics.")
    parser.add_argument(
        "--expert-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Base weight for one expert. Missing experts default to 1.0.",
    )
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-factor", type=float, default=0.15)
    parser.add_argument("--entropy-scale", type=float, default=1.0)
    parser.add_argument("--low-margin-threshold", type=float, default=0.15)
    parser.add_argument("--disagreement-penalty", type=float, default=0.55)
    return parser.parse_args()


def parse_weights(items: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--expert-weight must be NAME=WEIGHT, got {item!r}")
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def argmax(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda i: float(scores[i]))


def read_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    base_weights = parse_weights(args.expert_weight)
    router = OverlapAwareRouter(
        min_factor=args.min_factor,
        entropy_scale=args.entropy_scale,
        low_margin_threshold=args.low_margin_threshold,
        disagreement_penalty=args.disagreement_penalty,
    )

    rows = read_rows(args.input_jsonl)
    details = []
    raw_correct = 0
    fused_correct = 0
    teacher_correct = 0
    kl_before = []
    sym_kl = []
    factors = []

    for index, row in enumerate(rows):
        student_scores = [float(x) for x in row["student_scores"]]
        teacher_scores = [float(x) for x in row["teacher_scores"]]
        experts = [
            ExpertScoreDelta(
                name=name,
                scores=[float(x) for x in scores],
                base_weight=base_weights.get(name, 1.0),
            )
            for name, scores in row.get("experts", {}).items()
        ]
        fused_scores, weights, diagnostics = fuse_candidate_scores(
            student_scores,
            teacher_scores,
            experts,
            router=router,
            topk=args.topk,
            temperature=args.temperature,
        )
        target = int(row["target"])
        raw_pred = argmax(student_scores)
        fused_pred = argmax(fused_scores)
        teacher_pred = argmax(teacher_scores)
        raw_correct += int(raw_pred == target)
        fused_correct += int(fused_pred == target)
        teacher_correct += int(teacher_pred == target)
        kl_before.append(diagnostics.kl_teacher_student)
        sym_kl.append(diagnostics.symmetric_kl)
        factors.append(router.factor(diagnostics))
        details.append(
            {
                "id": row.get("id", row.get("doc_id", index)),
                "target": target,
                "raw_pred": raw_pred,
                "teacher_pred": teacher_pred,
                "fused_pred": fused_pred,
                "raw_correct": raw_pred == target,
                "teacher_correct": teacher_pred == target,
                "fused_correct": fused_pred == target,
                "expert_weights": weights,
                "diagnostics": diagnostics.__dict__,
                "fused_scores": fused_scores,
            }
        )

    n = max(len(rows), 1)
    summary = {
        "samples": len(rows),
        "raw_acc": raw_correct / n,
        "teacher_acc": teacher_correct / n,
        "fused_acc": fused_correct / n,
        "delta_acc": (fused_correct - raw_correct) / n,
        "mean_kl_teacher_student": mean(kl_before) if kl_before else None,
        "mean_symmetric_kl": mean(sym_kl) if sym_kl else None,
        "mean_router_factor": mean(factors) if factors else None,
        "expert_base_weights": base_weights,
        "router": {
            "topk": args.topk,
            "temperature": args.temperature,
            "min_factor": args.min_factor,
            "entropy_scale": args.entropy_scale,
            "low_margin_threshold": args.low_margin_threshold,
            "disagreement_penalty": args.disagreement_penalty,
        },
    }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
