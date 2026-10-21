#!/bin/bash
# AIME budget-fix rerun: same recipe/prompt/temp, max_out_len 32768 -> 65536
# (32768 truncated 4-6/30 items on every arm). Per arm: deploy(131072) -> run -> archive -> stop.
set -uo pipefail
LANE="${1:?lane}"
HOSTDIR=/home/yult/llm_infer_workspace
AB="$HOSTDIR/ylt/glm52-qad/aisbench"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PRODUCTS="${PRODUCTS:?}"
CFGSUF="${CFGSUF:-_hi}"   # e.g. _hi (temp 0.6/budget 65536) or _g0 (greedy)
LOG="$OUT/aimefix_${LANE}.log"
log(){ echo "[aimefix$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_arm(){
  local tag="$1" model="$2" mem="${3:-0.85}" maxlen="${4:-131072}"
  local ftag="${tag}${CFGSUF}"
  if [ -f "$OUT/aimefix_${ftag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 131072, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="front_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN="$maxlen" MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: AIME (official recipe + max_out_len 65536) ==="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD_HOST" \
    "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh ${ftag} 8000 ${ftag} aime2026_gen_0_shot_chat_prompt" >>"$LOG" 2>&1
  local rc=$?
  log "aime rc=$rc"
  if [ $rc -ne 0 ]; then
    log "$tag RUN FAILED rc=$rc (not marking DONE)"
    HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
    return 1
  fi
  local exp
  exp=$(ls -td "$AB"/outputs_${ftag}/*/ 2>/dev/null | head -1)
  mkdir -p "$OUT/aimefix_${ftag}_aisbench"
  cp -r "$exp"results/*/ "$OUT/aimefix_${ftag}_aisbench/" 2>/dev/null || true
  cp "$exp"summary/*.csv "$exp"summary/*.md "$OUT/aimefix_${ftag}_aisbench/" 2>/dev/null || true
  cp "$exp"predictions/vllm-api-general-chat/*.jsonl "$OUT/aimefix_${ftag}_aisbench/" 2>/dev/null || true
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/aimefix_${ftag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; rest2="${rest#*|}"
  mem="${rest2%%|*}"; maxlen="${rest2#*|}"
  [ "$mem" = "$model" ] && mem=0.85
  [ "$maxlen" = "$mem" ] && maxlen=131072
  run_arm "$tag" "$model" "$mem" "$maxlen"
done
touch "$OUT/AIMEFIX_${LANE}_DONE"
log "AIMEFIX LANE $LANE DONE"
