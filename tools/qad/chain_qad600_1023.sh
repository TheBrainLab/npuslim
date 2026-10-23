#!/bin/bash
# QAD600: MMLU-Pro 500 (non-thinking, 16K) + AIME 2026 (high, 131K/65K), 2-node TP=16.
# Same protocol as l0: mmlu_pro_probe.py (temp 0, cap 2048, 0-shot CoT, letter exact);
# AISBench aime2026_gen_0_shot_chat_prompt with temp 0.6, reasoning_effort=high, max_out 65536.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/mmlu_pro/mmlu_pro_500.parquet"
MODEL="/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_l0_clip_lo05_omni_qad600"
HEAD_IP="10.1.0.57"; WORKER_IP="10.1.0.47"
HEAD_HOST="yult@${HEAD_IP}"; WORKER_HOST="yult@${WORKER_IP}"
PORT=8000
LOG="$OUT/qad600_eval.log"
log(){ echo "[qad600 $(date '+%F %T')] $*" | tee -a "$LOG"; }
stop_deploy(){ HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; }

run_mmlupro(){
  if [ -f "$OUT/mmlupro_qad600_DONE" ]; then log "MMLU-Pro already done - skip"; return 0; fi
  log "=== MMLU-Pro deploy (maxlen 16384, mem 0.85) ==="
  stop_deploy; sleep 8
  MODEL="$MODEL" SERVED="mmp_qad600" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=16384 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "MMLU-Pro DEPLOY FAILED"; stop_deploy; return 1; fi
  log "=== MMLU-Pro 500 ==="
  python3 "$P2/mmlu_pro_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" \
    --model "mmp_qad600" --data "$DATA" --out "$OUT/mmlupro_qad600.jsonl" --workers 8 >>"$LOG" 2>&1
  local rc=$?
  log "mmlupro rc=$rc"
  stop_deploy
  if [ $rc -ne 0 ]; then log "MMLU-Pro RUN FAILED rc=$rc"; return 1; fi
  touch "$OUT/mmlupro_qad600_DONE"
  log "=== MMLU-Pro DONE ==="
}

run_aime(){
  if [ -f "$OUT/aimefix_qad600_re_DONE" ]; then log "AIME already done - skip"; return 0; fi
  log "=== AIME deploy (maxlen 131072, mem 0.85) ==="
  stop_deploy; sleep 8
  MODEL="$MODEL" SERVED="front_qad600" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=131072 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "AIME DEPLOY FAILED"; stop_deploy; return 1; fi
  log "=== AIME 2026 (high, 65536) ==="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD_HOST" \
    "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh qad600_re 8000 qad600_re aime2026_gen_0_shot_chat_prompt" >>"$LOG" 2>&1
  local rc=$?
  log "aime rc=$rc"
  if [ $rc -ne 0 ]; then log "AIME RUN FAILED rc=$rc"; stop_deploy; return 1; fi
  local exp
  exp=$(ls -td "$HOSTDIR/ylt/glm52-qad/aisbench/outputs_qad600_re"/*/ 2>/dev/null | head -1)
  mkdir -p "$OUT/aimefix_qad600_re_aisbench"
  cp -r "$exp"results/*/ "$OUT/aimefix_qad600_re_aisbench/" 2>/dev/null || true
  cp "$exp"summary/*.csv "$exp"summary/*.md "$OUT/aimefix_qad600_re_aisbench/" 2>/dev/null || true
  cp "$exp"predictions/vllm-api-general-chat/*.jsonl "$OUT/aimefix_qad600_re_aisbench/" 2>/dev/null || true
  stop_deploy
  touch "$OUT/aimefix_qad600_re_DONE"
  log "=== AIME DONE ==="
}

run_mmlupro || exit 1
run_aime || exit 1
touch "$OUT/QAD600_ALL_DONE"
log "=== QAD600 ALL DONE ==="
