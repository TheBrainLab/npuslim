#!/bin/bash
# run_he_vllm.sh — 在容器内用 lm_eval 跑 humaneval（并发 1 避免 aiohttp 崩溃）+ 截断重评
OUTDIR=/llm_infer_workspace/ylt/glm52-qad/eval_flipfiltered
MODELDIR=/llm_infer_workspace/ylt/glm52-qad/reencode_run14_flipfiltered

# code_eval patch (idempotent)
f=$(find /usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/humaneval -maxdepth 1 -name utils.py 2>/dev/null | head -1)
if [ -n "$f" ] && grep -q 'evaluate.load("code_eval")' "$f" 2>/dev/null; then
  cp "$f" "${f}.bak_$(date +%s)"
  sed -i 's|evaluate.load("code_eval")|evaluate.load("/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8/code_eval.py")|' "$f"
  echo "code_eval patched"
fi

echo "[$(date)] starting humaneval conc=1"
PYTHONUNBUFFERED=1 HF_ALLOW_CODE_EVAL=1 timeout 14400 lm_eval \
  --model local-completions \
  --model_args base_url=http://10.1.0.106:8000/v1/completions,tokenizer=$MODELDIR,num_concurrent=1,max_length=8192 \
  --tasks humaneval \
  --num_fewshot 0 \
  --gen_kwargs temperature=0 \
  --batch_size auto \
  --output_path $OUTDIR \
  --log_samples --confirm_run_unsafe_code \
  > $OUTDIR/he_lmeval3.log 2>&1
echo "LMEVAL_EXIT=$?" >> $OUTDIR/he_lmeval3.log

# trunc reeval
samp=$(ls -t ${OUTDIR}/qdpv68r5/samples_humaneval_*.jsonl 2>/dev/null | head -1)
if [ -n "$samp" ]; then
  HF_ALLOW_CODE_EVAL=1 PYTHONUNBUFFERED=1 python3 /llm_infer_workspace/ylt/npuslim/workspace/glm5.2_deploy/reeval_humaneval_current.py flipfiltered_lmeval=${samp} >> $OUTDIR/he_lmeval3.log 2>&1
fi
echo "ALL DONE $(date)"
