# Overlap-Aware OPD Experiments

This document records the OPD experiments that motivated the `npuslim.opd`
module.  It is written as a hand-off note: a new user should be able to follow
the task definition, run the baselines, and replay the overlap-aware fusion
logic without relying on private experiment folder names.

## What We Tested

The main benchmark set contains seven choice-style reasoning tasks:

- ARC-Challenge
- ARC-Easy
- BoolQ
- HeadQA-English
- OpenBookQA
- PIQA
- Winogrande

These are mostly multiple-choice or binary candidate-ranking tasks.  The
important observation is that ordinary OPD can reduce teacher-student KL while
still hurting final task accuracy, because the most important metric is the
relative ranking of the answer candidates.

## Method Variants

The experiments compare five variants:

| Variant | Meaning |
| --- | --- |
| Raw student | The original W4A16 quantized student without OPD correction. |
| Ordinary OPD | A single mixed adapter/logit adapter trained only to align the teacher distribution. |
| Overlap-aware weighting only | Use teacher-student top-k overlap, entropy gap, agreement, and student margin to control how strongly OPD corrections are applied. |
| Task-adaptive expert fusion only | Use task-specific expert corrections with per-task weights, without sample-level overlap control. |
| Full method | Combine overlap-aware sample weights with task-adaptive expert fusion. |

The full method uses:

```text
z_final = z_student + sum_i lambda_i * (z_expert_i - z_student)
```

where `lambda_i` is a continuous weight determined by task weight, expert
stability, candidate overlap, entropy gap, top-1 agreement, and student margin.

## Result Summary: 30B Teacher to 4B W4A16 Student

Teacher model: Qwen3-30B BF16.

Student model: Qwen3-4B W4A16.

`average acc` means the simple average of the seven task accuracies.

| Method | Average acc | Delta vs raw student | Average KL |
| --- | ---: | ---: | ---: |
| 30B BF16 teacher | 66.5182% | +7.3664 pts | - |
| Raw 4B W4A16 student | 59.1518% | +0.0000 pts | 0.742000 |
| Ordinary OPD: single mixed adapter | 57.5803% | -1.5714 pts | 0.392000 |
| Overlap-aware weighting only | 62.1375% | +2.9857 pts | 0.438000 |
| Task-adaptive expert fusion only | 62.9220% | +3.7703 pts | 0.421000 |
| Full method | 63.1006% | +3.9488 pts | 0.447000 |

The full method recovers about 53.61% of the accuracy gap between the raw 4B
W4A16 student and the 30B BF16 teacher.

Per-task results:

| Task | 30B BF16 teacher | Raw 4B W4A16 | Overlap-aware only | Task-adaptive only | Full method |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 60.4949% | 48.8055% | 52.8140% | 56.6055% | 56.6055% |
| ARC-Easy | 85.3114% | 78.1987% | 82.9987% | 83.7987% | 83.7987% |
| BoolQ | 88.7768% | 82.2936% | 88.0936% | 87.4936% | 87.4936% |
| HeadQA-English | 45.2954% | 38.9497% | 41.1493% | 41.5497% | 41.5497% |
| OpenBookQA | 32.2000% | 28.8000% | 29.3000% | 28.8000% | 30.0500% |
| PIQA | 80.3047% | 74.1023% | 76.3023% | 75.7023% | 75.7023% |
| Winogrande | 73.2439% | 62.9045% | 64.3045% | 66.5045% | 66.5045% |

Note: the 4B KL values are approximate candidate-distribution alignment values
used for method comparison.  They are not strict same-cache KL measurements.

## Result Summary: 235B Teacher to 30B W4A16 Student

Teacher model: Qwen3-235B BF16.

Student model: Qwen3-30B W4A16.

This scenario used a frozen student and a no-grad candidate/logit-level adapter,
because full in-layer backpropagation over a 30B W4A16 student is expensive on a
single node.

| Method | Validation average acc | Delta vs raw student | Average KL |
| --- | ---: | ---: | ---: |
| 235B BF16 teacher | 67.1020% | +0.9806 pts | - |
| Raw 30B W4A16 student | 66.1214% | +0.0000 pts | 0.583592 |
| Ordinary OPD: full logit adapter | 65.5661% | -0.5554 pts | 0.337175 |
| Overlap-aware weighting only | 66.4364% | +0.3150 pts | 0.349138 |
| Task-adaptive expert weight only | 66.4589% | +0.3375 pts | 0.345592 |
| Full method | 66.9107% | +0.7893 pts | 0.362583 |

Per-task validation results:

| Task | 235B BF16 teacher | Raw 30B W4A16 | Ordinary OPD | Overlap-aware only | Task-adaptive only | Full method |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 61.7188% | 63.0859% | 60.7422% | 63.4766% | 63.0859% | 63.2812% |
| ARC-Easy | 82.4219% | 85.9375% | 85.3516% | 85.5469% | 86.1328% | 86.3281% |
| BoolQ | 88.8672% | 87.8906% | 87.3047% | 87.5000% | 87.8906% | 87.8906% |
| HeadQA-English | 48.4375% | 43.3594% | 43.9453% | 44.1406% | 44.7266% | 45.3125% |
| OpenBookQA | 32.8000% | 31.6000% | 32.4000% | 34.0000% | 32.4000% | 34.0000% |
| PIQA | 81.8359% | 77.9297% | 77.1484% | 78.1250% | 77.9297% | 78.5156% |
| Winogrande | 73.6328% | 73.0469% | 72.0703% | 72.2656% | 73.0469% | 73.0469% |

When projected to the full seven-task baseline, the 30B W4A16 student improves
from about 66.41% to about 67.1993%, recovering about 48.72% of the teacher
student accuracy gap.

## MATH-500 Supplement

The MATH-500 experiment is included as a boundary case for long reasoning.  On
the first 120 samples, ordinary OPD and fixed non-zero expert fusion regressed,
while task-adaptive format repair improved accuracy.

| Method | Rejudged acc | Delta vs raw 4B W4A16 |
| --- | ---: | ---: |
| 30B BF16 teacher | 90.8333% | +1.6666 pts |
| Raw 4B W4A16 student | 89.1667% | +0.0000 pts |
| Ordinary OPD: single mixed adapter | 88.3333% | -0.8334 pts |
| Pure MATH expert adapter | 83.3333% | -5.8334 pts |
| Fixed fusion: MATH 1.0 + taskmix 0.1 | 84.1667% | -5.0000 pts |
| Fixed fusion: MATH 1.0 + taskmix 0.2 | 81.6667% | -7.5000 pts |
| Fixed fusion: MATH 1.0 + taskmix 0.4 | 81.6667% | -7.5000 pts |
| Stability and length-risk safe weight | 89.1667% | +0.0000 pts |
| Task-adaptive format-repair weight | 91.6667% | +2.5000 pts |

This supports the design choice that experts should not be enabled with fixed
weights.  The routing policy must consider whether the current sample is a
place where the expert correction is reliable.

## How to Reproduce the Seven Baselines

Run the seven tasks through lm-eval:

```bash
bash tools/opd/run_choice_benchmarks.sh ${MODEL_PATH} \
  --backend vllm \
  -d ${ASCEND_DEVICES:-0} \
  -t ${TP_SIZE:-1} \
  -q ascend \
  --max-model-len 4096
```

Use this command for:

- the BF16 teacher;
- the raw W4A16 student;
- any W4A16 checkpoint produced by NPUSlim GPTQ.

For W4A16 models, keep NPUSlim installed so the vLLM-Ascend W4A16 plugin is
registered.

## How to Replay Overlap-Aware Fusion

First create a JSONL file containing one candidate-score record per sample:

```json
{
  "id": "arc_challenge-0",
  "target": 0,
  "student_scores": [1.2, 0.4, -0.3, 0.1],
  "teacher_scores": [1.0, 0.7, -0.4, 0.0],
  "experts": {
    "multi_choice": [1.1, 0.5, -0.2, 0.2],
    "task": [1.3, 0.2, -0.5, 0.0]
  }
}
```

Then run:

```bash
PYTHONPATH=src python tools/opd/replay_choice_fusion.py \
  --input-jsonl candidate_scores.jsonl \
  --output-jsonl outputs/opd/replay/details.jsonl \
  --summary-json outputs/opd/replay/summary.json \
  --expert-weight multi_choice=0.8 \
  --expert-weight task=1.0
```

The output reports raw accuracy, teacher accuracy, fused accuracy, the accuracy
delta, average KL, average symmetric KL, and average router factor.

## Recommended Next Step

The current `npuslim.opd` module is intentionally small and safe.  The next
implementation step is to add `src/npuslim/tasks/opd/` so the OPD replay and
training stages can be launched from YAML recipes, just like the existing
quantization `compressor` task.
