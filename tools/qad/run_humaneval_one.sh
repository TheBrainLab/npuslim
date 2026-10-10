#!/bin/bash
# run_humaneval_one.sh — 单产物 humaneval（0-shot conc=4 maxlen=8192 greedy）+ 截断重评
# 用法（在容器内）: bash run_humaneval_one.sh <model_dir> <out_dir> <tag>
set -u
MODEL_DIR="$1"
OUT="$2"
TAG="$3"
BASE_URL="http://10.1.0.106:8000/v1/completions"
CODEEVAL_DIR="/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8"

mkdir -p "$OUT"

# code_eval offline patch (idempotent)
f=$(find /usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/humaneval -maxdepth 1 -name utils.py 2>/dev/null | head -1)
if [ -n "$f" ] && ! grep -q "qwen3_30b_w4a8/code_eval.py" "$f"; then
  cp "$f" "${f}.bak_$(date +%s)"
  sed -i "s|evaluate.load(\"code_eval\")|evaluate.load(\"${CODEEVAL_DIR}/code_eval.py\")|" "$f"
  echo "code_eval patched"
fi

echo "[$(date '+%F %T')] humaneval start: $TAG model=$MODEL_DIR"
HF_ALLOW_CODE_EVAL=1 PYTHONUNBUFFERED=1 lm_eval \
  --model local-completions \
  --model_args base_url=${BASE_URL},tokenizer=${MODEL_DIR},num_concurrent=4,max_length=8192 \
  --tasks humaneval \
  --num_fewshot 0 \
  --gen_kwargs temperature=0 \
  --batch_size auto \
  --output_path "$OUT" \
  --log_samples \
  --confirm_run_unsafe_code \
  > "$OUT/he_${TAG}.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] humaneval done rc=$rc"
# 截断重评
samples=$(ls -t "$OUT"/*/samples_humaneval_*.jsonl 2>/dev/null | head -1)
if [ -n "$samples" ]; then
  echo "samples: $samples"
  HF_ALLOW_CODE_EVAL=1 python3 /llm_infer_workspace/ylt/npuslim/workspace/glm5.2_deploy/reeval_humaneval_current.py "${TAG}=${samples}"
fi
