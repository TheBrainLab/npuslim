#!/bin/bash
# Wait for HLE-MC cap pilot DONE, then run 64K/128K multiquery 3 arms:
#   F lane l0 (2 nodes), G lane w4a8c8 (2 nodes), then BF16 (4 nodes).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
LOG="$OUT/mq_all.log"
log(){ echo "[mqall $(date '+%F %T')] $*" | tee -a "$LOG"; }
W4A8="/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq"
W8="/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq"

CAP_PID=$(cat "$OUT/hle_mc_cap.pid" 2>/dev/null || true)
deadline=$(( $(date +%s) + 3*3600 ))
while [ ! -f "$OUT/hle_mc_cap_DONE" ]; do
  if [ -n "$CAP_PID" ] && ! kill -0 "$CAP_PID" 2>/dev/null; then
    log "HLE cap pilot $CAP_PID died without DONE; not starting multiquery"
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "HLE cap pilot timeout; not starting multiquery"
    exit 1
  fi
  sleep 60
done
log "HLE cap pilot DONE; starting multiquery 64K/128K (l0/w4a8c8 + BF16)"

HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 PRODUCTS="l0|${W4A8}/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85" \
  setsid nohup bash "$QAD/chain_multiquery_1019.sh" F >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/mq_F.pid"
HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 PRODUCTS="p2clip|${W4A8}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w8a8|${W8}/glm5-w8a8_replica_p2clipfold_w8expert|0.95" \
  setsid nohup bash "$QAD/chain_multiquery_1019.sh" G >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/mq_G.pid"

for lane in F G; do
  deadline=$(( $(date +%s) + 4*3600 ))
  while [ ! -f "$OUT/MQ_${lane}_DONE" ]; do
    pid=$(cat "$OUT/mq_${lane}.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "lane $lane process $pid died without DONE"; exit 1
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "lane $lane timeout"; exit 1
    fi
    sleep 60
  done
  log "lane $lane DONE"
done

log "starting BF16 multiquery (4 nodes)"
setsid nohup bash "$QAD/bf16_multiquery_1019.sh" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/mq_bf16.pid"
deadline=$(( $(date +%s) + 4*3600 ))
while [ ! -f "$OUT/mq_bf16_DONE" ]; do
  pid=$(cat "$OUT/mq_bf16.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    log "BF16 process $pid died without DONE"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "BF16 timeout"; exit 1
  fi
  sleep 60
done
touch "$OUT/MQ_ALL_DONE"
log "ALL MULTIQUERY DONE"
