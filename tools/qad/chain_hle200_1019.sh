#!/bin/bash
# HLE-MC 200 questions at cap 24576, per lane (2 nodes). 2 arms: l0/w4a8c8.
set -uo pipefail
LANE="${1:?lane}"
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_200.parquet"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?}"
CAP="${CAP:-24576}"
MAXLEN="${MAXLEN:-40960}"
LOG="$OUT/hle200_${LANE}.log"
log(){ echo "[hle200$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
stop_deploy(){ HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; }

run_arm(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/hle200_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen $MAXLEN, cap $CAP, mem $mem) ==="
  stop_deploy; sleep 8
  MODEL="$model" SERVED="hle200_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN="$MAXLEN" MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; stop_deploy; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: HLE-MC 200 (cap $CAP) ==="
  python3 "$P2/hle_mc_probe.py" --api "$api" --model "hle200_${tag}" --data "$DATA" \
    --out "$OUT/hle200_${tag}.jsonl" --workers 4 --max-tokens "$CAP" \
    --enable-thinking --reasoning-effort high --temperature 0.6 >>"$LOG" 2>&1
  local rc=$?
  log "hle200 rc=$rc"
  stop_deploy
  if [ $rc -ne 0 ]; then log "$tag RUN FAILED rc=$rc (not marking DONE)"; return 1; fi
  touch "$OUT/hle200_${tag}_DONE"
  log "=== $tag HLE 200 DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_arm "$tag" "$model" "$mem" || exit 1
done
touch "$OUT/HLE200_${LANE}_DONE"
log "HLE200 LANE $LANE DONE"
