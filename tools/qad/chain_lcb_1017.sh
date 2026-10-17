#!/bin/bash
# LiveCodeBench v6 (code_generation_lite, 1055 problems) cross-validation round (1017).
# Per product: deploy (TP=16, maxlen 8192) -> lcb_probe (gen) -> stop -> lcb_score (CPU).
set -uo pipefail
LANE="${1:-${LANE:-LCB}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/lcb_${LANE}.log"
log(){ echo "[lcb$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/lcb_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 8192, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="lcb_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=8192 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: LCB generation (1055 problems) ==="
  python3 "$P2/lcb_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "lcb_${tag}" \
    --out "$OUT/lcb_${tag}_gen.jsonl" --workers 8 >>"$LOG" 2>&1
  log "gen rc=$?"
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  log "=== $tag: LCB scoring (official vendored) ==="
  python3 "$P2/lcb_score.py" --gen "$OUT/lcb_${tag}_gen.jsonl" --out "$OUT/lcb_${tag}_summary.json" --workers 24 >>"$LOG" 2>&1
  log "score rc=$?"
  touch "$OUT/lcb_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/LCB_${LANE}_DONE"
log "LCB LANE $LANE DONE"
