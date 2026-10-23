#!/bin/bash
# MMLU-Pro 500 (5 arms) + optional l0 HLE smoke, per lane (2 nodes).
# Lane F: l0 -> HLE smoke -> w4a8c8 ; Lane G: p2clip -> w8a8.
set -uo pipefail
LANE="${1:?lane}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/mmlu_pro/mmlu_pro_500.parquet"
HLE_DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_smoke20.parquet"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?}"
HLE_AFTER_L0="${HLE_AFTER_L0:-0}"
MAXLEN="${MAXLEN:-16384}"
LOG="$OUT/mmlupro_${LANE}.log"
log(){ echo "[mmp$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
stop_deploy(){ HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; }

run_mmlupro(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/mmlupro_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen $MAXLEN, mem $mem) ==="
  stop_deploy; sleep 8
  MODEL="$model" SERVED="mmp_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN="$MAXLEN" MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; stop_deploy; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: MMLU-Pro 500 ==="
  python3 "$P2/mmlu_pro_probe.py" --api "$api" --model "mmp_${tag}" --data "$DATA" \
    --out "$OUT/mmlupro_${tag}.jsonl" --workers 8 >>"$LOG" 2>&1
  local rc=$?
  log "mmlupro rc=$rc"
  stop_deploy
  if [ $rc -ne 0 ]; then log "$tag RUN FAILED rc=$rc (not marking DONE)"; return 1; fi
  touch "$OUT/mmlupro_${tag}_DONE"
  log "=== $tag MMLU-Pro DONE ==="
}

run_hle_smoke(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/hle_smoke20_${tag}_DONE" ]; then log "$tag HLE smoke already done - skip"; return 0; fi
  log "=== $tag: deploy for HLE smoke (maxlen 69632, mem $mem) ==="
  stop_deploy; sleep 8
  MODEL="$model" SERVED="hle_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=69632 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag HLE DEPLOY FAILED"; stop_deploy; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: HLE smoke20 ==="
  python3 "$P2/hle_probe.py" --api "$api" --model "hle_${tag}" \
    --data "$HLE_DATA" --out "$OUT/hle_smoke20_${tag}.jsonl" \
    --workers 2 --max-tokens 32768 >>"$LOG" 2>&1
  local rc=$?
  log "hle smoke rc=$rc"
  stop_deploy
  if [ $rc -ne 0 ]; then log "$tag HLE RUN FAILED rc=$rc"; return 1; fi
  touch "$OUT/hle_smoke20_${tag}_DONE"
  log "=== $tag HLE smoke DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_mmlupro "$tag" "$model" "$mem" || exit 1
  if [ "$tag" = "l0" ] && [ "$HLE_AFTER_L0" = "1" ]; then
    run_hle_smoke "$tag" "$model" "$mem" || exit 1
  fi
done
touch "$OUT/MMLUPRO_${LANE}_DONE"
log "MMLUPRO LANE $LANE DONE"
