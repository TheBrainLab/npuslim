# Overlap-Aware OPD Integration

This note describes how the patent-facing OPD work should enter NPUSlim.

## Scope

The OPD module should not duplicate NPUSlim's quantization and vLLM-Ascend
deployment code. It should consume candidate scores or logits produced by a
quantized student and teacher, then return:

- candidate-level diagnostics, including top-k overlap, entropy gap, margin,
  agreement, and KL;
- continuous expert weights;
- fused candidate scores:
  `z_final = z_student + sum_i lambda_i * (z_expert_i - z_student)`.

## Recommended Layout

- `src/npuslim/opd/diagnostics.py`: candidate diagnostics and KL utilities.
- `src/npuslim/opd/fusion.py`: overlap-aware router and continuous expert fusion.
- `src/npuslim/opd/calibrator.py`: lightweight trainable candidate-score adapter.
- Future `src/npuslim/tasks/opd/`: config-driven OPD training/evaluation task.
- Future `tools/opd/`: runnable scripts for seven choice-style benchmarks and
  MATH-style trajectory evaluation.

## Mapping From Local Experiments

- `opd_train_dense_qwen3_adapter.py`: keep the LoRA adapter idea, but reuse
  NPUSlim/vLLM-Ascend W4A16 loading where possible.
- `opd_train_candidate_adapter.py`: migrate candidate sample parsing and
  teacher-distribution KL objective into a future OPD task.
- `opd_train_candidate_logit_adapter.py`: migrate the score-level calibrator
  into `npuslim.opd.calibrator`.
- `opd_eval_dense_qwen3_expert_fusion_mc.py`: migrate its continuous fusion
  equation into `npuslim.opd.fusion`.
- `scripts/analyze_overlap_aware_score_fusion.py`: migrate overlap, entropy,
  margin, and agreement diagnostics into `npuslim.opd.diagnostics`.

## Integration Strategy

1. Keep the current commit small: add reusable OPD primitives only.
2. Add a config-driven `opd` task after confirming the remote NPU runtime
   imports NPUSlim, vLLM-Ascend, Transformers, and lm-eval together.
3. Preserve the existing `compressor` task for quantization. OPD should run
   after a W4A16 checkpoint already exists, or operate on candidate-score
   artifacts emitted by evaluation tools.
4. For 235B teacher to 30B W4A16 student, prefer score/logit-level adapters
   because full in-layer training can exceed a single node's memory budget.
