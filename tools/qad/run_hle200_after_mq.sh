#!/bin/bash
# Wait for 5-arm multiquery DONE, then run HLE-MC 200 for l0 + w4a8c8 (2 lanes).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
LOG="$OUT/hle200_all.log"
log(){ echo "[hle200all $(date '+%F %T')] $*" | tee -a "$LOG"; }
W4A8="/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq"

deadline=$(( $(date +%s) + 8*3600 ))
while [ ! -f "$OUT/MQ_ALL_DONE" ]; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "multiquery timeout; not starting HLE 200"
    exit 1
  fi
  sleep 60
done
log "multiquery DONE; starting HLE 200 (l0 + w4a8c8, cap 24576)"

HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 PRODUCTS="l0|${W4A8}/glm5-w4a8_replica_l0_clip_lo05_omni|0.85" \
  setsid nohup bash "$QAD/chain_hle200_1019.sh" F >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/hle200_F.pid"
HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 PRODUCTS="w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85" \
  setsid nohup bash "$QAD/chain_hle200_1019.sh" G >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/hle200_G.pid"

for lane in F G; do
  deadline=$(( $(date +%s) + 14*3600 ))
  while [ ! -f "$OUT/HLE200_${lane}_DONE" ]; do
    pid=$(cat "$OUT/hle200_${lane}.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "lane $lane process $pid died without DONE"; exit 1
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "lane $lane timeout"; exit 1
    fi
    sleep 120
  done
  log "lane $lane DONE"
done
touch "$OUT/HLE200_ALL_DONE"
log "ALL HLE 200 DONE"
