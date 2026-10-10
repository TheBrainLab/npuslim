#!/bin/bash
# run_eval_in_container.sh — 在容器内跑四维评测（绕开宿主机无 docker 组权限）
set -e
OUTDIR=/llm_infer_workspace/ylt/glm52-qad/eval_run14_step600
mkdir -p "$OUTDIR"

echo "=== round 1/3: wikitext ppl ==="
PYTHONUNBUFFERED=1 lm_eval \
  --model local-completions \
  --model_args base_url=http://10.1.0.106:8000/v1/completions,tokenizer=/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600,num_concurrent=2,max_length=1024 \
  --tasks wikitext \
  --num_fewshot 0 \
  --gen_kwargs temperature=0 \
  --batch_size auto \
  --output_path "$OUTDIR" \
  --log_samples 2>&1 | tee "$OUTDIR/round_wikitext.log"
rc=$?
echo "round 1 exit=$rc"
[ $rc -eq 0 ] || exit $rc

echo "=== round 2/3: humaneval 0-shot ==="
PYTHONUNBUFFERED=1 lm_eval \
  --model local-completions \
  --model_args base_url=http://10.1.0.106:8000/v1/completions,tokenizer=/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600,num_concurrent=8,max_length=8192 \
  --tasks humaneval \
  --num_fewshot 0 \
  --gen_kwargs temperature=0 \
  --batch_size auto \
  --output_path "$OUTDIR" \
  --log_samples --confirm_run_unsafe_code 2>&1 | tee "$OUTDIR/round_humaneval.log"
rc=$?
echo "round 2 exit=$rc"
[ $rc -eq 0 ] || exit $rc

echo "=== round 3/3: mmlu + gsm8k 5-shot ==="
PYTHONUNBUFFERED=1 lm_eval \
  --model local-completions \
  --model_args base_url=http://10.1.0.106:8000/v1/completions,tokenizer=/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600,num_concurrent=64,max_length=8192 \
  --tasks mmlu,gsm8k \
  --num_fewshot 5 \
  --gen_kwargs temperature=0 \
  --batch_size auto \
  --output_path "$OUTDIR" \
  --log_samples 2>&1 | tee "$OUTDIR/round_mmlu_gsm8k.log"
rc=$?
echo "round 3 exit=$rc"

echo "=== ALL DONE ==="
