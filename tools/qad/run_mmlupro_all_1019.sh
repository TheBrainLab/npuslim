#!/bin/bash
# Orchestrate: F lane (l0 + HLE smoke + w4a8c8), G lane (p2clip + w8a8),
# then BF16 (4 nodes). All detached; logs under glm52-qad/audit1009/.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
CHAIN="$QAD/chain_mmlupro_1019.sh"
BF16_CHAIN="$QAD/bf16_mmlupro_1019.sh"
LOG="$OUT/mmlupro_all.log"
log(){ echo "[mmpall $(date '+%F %T')] $*" | tee -a "$LOG"; }
W4A8="/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq"
W8="/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq"

F_PRODUCTS="l0|${W4A8}/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85"
G_PRODUCTS="p2clip|${W4A8}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w8a8|${W8}/glm5-w8a8_replica_p2clipfold_w8expert|0.95"

start_lane(){
  local lane="$1" head="$2" worker="$3" products="$4" hle="$5"
  log "start lane $lane (head=$head worker=$worker hle=$hle)"
  HEAD_IP="$head" WORKER_IP="$worker" PRODUCTS="$products" HLE_AFTER_L0="$hle" \
    setsid nohup bash "$CHAIN" "$lane" >>"$LOG" 2>&1 < /dev/null &
  echo $! > "$OUT/mmlupro_${lane}.pid"
}

wait_lane(){
  local lane="$1" deadline=$(( $(date +%s) + 8*3600 ))
  while [ ! -f "$OUT/MMLUPRO_${lane}_DONE" ]; do
    local pid; pid=$(cat "$OUT/mmlupro_${lane}.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "lane $lane process $pid died without DONE"; return 1
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "lane $lane timeout"; return 1
    fi
    sleep 60
  done
  log "lane $lane DONE"
}

start_lane F 10.1.0.57 10.1.0.47 "$F_PRODUCTS" 1
start_lane G 10.1.0.106 10.1.0.135 "$G_PRODUCTS" 0
wait_lane F || exit 1
wait_lane G || exit 1

log "=== all W4A8 lanes done; starting BF16 (4 nodes) ==="
setsid nohup bash "$BF16_CHAIN" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/mmlupro_bf16.pid"
deadline=$(( $(date +%s) + 4*3600 ))
while [ ! -f "$OUT/mmlupro_bf16_DONE" ]; do
  pid=$(cat "$OUT/mmlupro_bf16.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    log "BF16 process $pid died without DONE"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "BF16 timeout"; exit 1
  fi
  sleep 60
done
touch "$OUT/MMLUPRO_ALL_DONE"
log "ALL MMLU-PRO DONE (F/G + BF16; HLE smoke on F lane if enabled)"
