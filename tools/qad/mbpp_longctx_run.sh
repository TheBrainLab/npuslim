#!/bin/bash
# =============================================================================
# mbpp_longctx_run.sh — P2a: long-context code eval (single product, single round).
#   deploy (TP=16, .57+.47) -> health -> lm_eval custom task `mbpp_longctx`
#   (32K-token prefix + original mbpp prompt, same tests/extraction as mbpp) -> stop.
#
# Usage (on 10.1.0.106):
#   TAG=<tag> MODEL_DIR=<container artifact path> \
#     setsid nohup bash tools/qad/mbpp_longctx_run.sh >/tmp/<tag>.lctx.log 2>&1 &
# =============================================================================
set -uo pipefail
TAG="${TAG:?}"; MODEL_DIR="${MODEL_DIR:?}"
HEAD_HOST="${HEAD_HOST:-yult@10.1.0.106}"; WORKER_HOST="${WORKER_HOST:-yult@10.1.0.57}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT_C="/llm_infer_workspace/ylt/glm52-qad/audit1009/${TAG}"
CONTAINER=e6_noawq_env
PORT=8000
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx
MAXLEN="${MAXLEN:-36864}"
CONC="${CONC:-4}"
MEMUTIL="${MEMUTIL:-0.85}"
LOG="$HOSTDIR/ylt/glm52-qad/audit1009/${TAG}.lctx.log"
say(){ echo "[$TAG $(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=========== LONGCTX MBPP START model=$MODEL_DIR ==========="
MODEL="$MODEL_DIR" SERVED="lctx_$TAG" \
  MAX_MODEL_LEN="$MAXLEN" MEM_UTIL="$MEMUTIL" MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS="$CONC" \
  bash "$DEPLOY" >>"$LOG" 2>&1
say "deploy rc=$?"

ok=0
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:${PORT}/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then ok=1; say "health 200 (${i} x 10s)"; break; fi
  sleep 10
done
if [ "$ok" != 1 ]; then say "HEALTH TIMEOUT - abort+stop"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi

docker exec \
  -w "$TASK_DIR" \
  -e HF_HOME=/llm_infer_workspace/hf_home \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OPENAI_API_KEY=EMPTY -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e VLLM_RPC_TIMEOUT=360000 -e HF_ALLOW_CODE_EVAL=1 \
  -e no_proxy=localhost,127.0.0.1 -e NO_PROXY=localhost,127.0.0.1 \
  "$CONTAINER" bash -lc "PYTHONUNBUFFERED=1 lm_eval \
    --model local-completions \
    --model_args base_url=http://localhost:${PORT}/v1/completions,tokenizer=${MODEL_DIR},num_concurrent=${CONC},max_length=${MAXLEN} \
    --tasks mbpp_longctx \
    --include_path ${TASK_DIR} \
    --num_fewshot 0 \
    --gen_kwargs temperature=0 \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --output_path ${OUT_C} \
    --log_samples" >>"$LOG" 2>&1
say "lm_eval rc=$?"

bash "$DEPLOY" --stop >>"$LOG" 2>&1 || true
say "=========== LONGCTX MBPP DONE ==========="
