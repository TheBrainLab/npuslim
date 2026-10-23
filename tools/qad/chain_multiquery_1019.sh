#!/bin/bash
# 64K/128K multiquery NIAH, per lane (2 nodes), 36 cases/arm.
set -uo pipefail
LANE="${1:?lane}"
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?}"
MAXLEN="${MAXLEN:-135168}"
LOG="$OUT/mq_${LANE}.log"
log(){ echo "[mq$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
stop_deploy(){ HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; }

run_arm(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/mq_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen $MAXLEN, mem $mem) ==="
  stop_deploy; sleep 8
  MODEL="$model" SERVED="mq_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN="$MAXLEN" MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; stop_deploy; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: multiquery 64K/128K ==="
  python3 "$P2/niah_multiquery_probe.py" --api "$api" --model "mq_${tag}" \
    --out "$OUT/mq_${tag}.jsonl" --lengths 65536,131072 --per-cell 6 --workers 2 \
    >>"$LOG" 2>&1
  local rc=$?
  log "multiquery rc=$rc"
  stop_deploy
  if [ $rc -ne 0 ]; then log "$tag RUN FAILED rc=$rc (not marking DONE)"; return 1; fi
  touch "$OUT/mq_${tag}_DONE"
  log "=== $tag multiquery DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_arm "$tag" "$model" "$mem" || exit 1
done
touch "$OUT/MQ_${LANE}_DONE"
log "MQ LANE $LANE DONE"
