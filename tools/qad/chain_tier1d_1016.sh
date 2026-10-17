#!/bin/bash
# Tier1d: RULER-hard v2 re-run (genuine var_track_chain, 64K) for 4 products.
set -uo pipefail
LANE="${1:-${LANE:-E}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/tier1d_${LANE}.log"
log(){ echo "[t1d$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/t1d_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 69632, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="t1d_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=69632 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip probe"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: RULER-hard v2 (var_track_chain) ==="
  python3 "$P2/ruler_hard_probe_v2.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1d_${tag}" \
    --out "$OUT/t1d_${tag}_ruler2.jsonl" >>"$LOG" 2>&1
  log "ruler2 rc=$?"
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/t1d_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/T1D_${LANE}_DONE"
log "TIER1D LANE $LANE DONE"
