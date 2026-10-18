#!/bin/bash
# cap-corrected full re-run (1018 audit): simpleqa + ifbench @ cap 1024.
set -uo pipefail
LANE="${1:-${LANE:-CF}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/capfix_${LANE}.log"
log(){ echo "[capfix$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/capfix_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 8192, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="capfix_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=8192 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: simpleqa (cap 1024) ==="
  python3 "$P2/simpleqa_probe_c1024.py" --api "$api" --model "capfix_${tag}" --out "$OUT/capfix_${tag}_simpleqa.jsonl" --workers 16 >>"$LOG" 2>&1
  log "simpleqa rc=$?"
  log "=== $tag: ifbench (cap 1024) ==="
  python3 "$P2/ifbench_probe_c1024.py" --api "$api" --model "capfix_${tag}" --out "$OUT/capfix_${tag}_ifbench.jsonl" --workers 16 >>"$LOG" 2>&1
  log "ifbench rc=$?"
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/capfix_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/CAPFIX_${LANE}_DONE"
log "CAPFIX LANE $LANE DONE"
