#!/bin/bash
# =============================================================================
# watch_final_1019.sh — watchdog for the final 1019 rounds:
#   frontier lanes F/G (w4a8c8/w8a8) -> BF16 4-node arm -> GPQA dedicated round.
#   Every 10 min: if a lane is not DONE and its driver process is gone -> relaunch
#   (all chains are idempotent: per-arm *_DONE markers skip completed arms).
#   Also logs WARN when a live lane's log has been stale > 5 h (no action).
#   Pattern note (AGENTS #14): anchored patterns avoid matching the old launcher
#   bash -c cmdline; pgrep never matches itself.
# =============================================================================
set -uo pipefail
OUT=/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009
REPO=/home/yult/llm_infer_workspace/ylt/npuslim_repo/tools/qad
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
W8=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq
WLOG="$OUT/watch_final.log"
log(){ echo "[watch $(date '+%F %T')] $*" | tee -a "$WLOG"; }

launch_F(){
  env LANE=F HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47 \
    PRODUCTS="l0|$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85" \
    setsid nohup bash "$REPO/chain_frontier_1019.sh" F >>"$OUT/frontier_F.log" 2>&1 < /dev/null &
}
launch_G(){
  env LANE=G HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
    PRODUCTS="p2clip|$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w8a8|$W8/glm5-w8a8_replica_p2clipfold_w8expert|0.95" \
    setsid nohup bash "$REPO/chain_frontier_1019.sh" G >>"$OUT/frontier_G.log" 2>&1 < /dev/null &
}
launch_bf16(){ setsid nohup bash "$REPO/bf16_frontier_1019.sh" >>"$OUT/frontier_bf16.log" 2>&1 < /dev/null & }
launch_gpqa(){ setsid nohup bash "$REPO/gpqa_launch_1019.sh" >>"$OUT/gpqa_launch.log" 2>&1 < /dev/null & }

stale_h(){ local f="$1"; [ -f "$f" ] || { echo 0; return; }
  local m; m=$(stat -c %Y "$f"); echo $(( ( $(date +%s) - m ) / 3600 )); }

log "watchdog start (pid $$)"
for it in $(seq 1 288); do
  if [ ! -f "$OUT/FRONTIER_F_DONE" ]; then
    if ! pgrep -f 'chain_frontier_1019.sh F$' >/dev/null 2>&1; then log "F driver gone -> relaunch"; launch_F; fi
    s=$(stale_h "$OUT/frontier_F.log"); [ "$s" -gt 5 ] && log "WARN F log stale ${s}h"
  fi
  if [ ! -f "$OUT/FRONTIER_G_DONE" ]; then
    if ! pgrep -f 'chain_frontier_1019.sh G$' >/dev/null 2>&1; then log "G driver gone -> relaunch"; launch_G; fi
    s=$(stale_h "$OUT/frontier_G.log"); [ "$s" -gt 5 ] && log "WARN G log stale ${s}h"
  fi
  if [ -f "$OUT/FRONTIER_F_DONE" ] && [ -f "$OUT/FRONTIER_G_DONE" ] && [ ! -f "$OUT/frontier_bf16_DONE" ]; then
    if ! pgrep -f 'bf16_frontier_1019.sh$' >/dev/null 2>&1; then log "BF16 driver gone -> relaunch"; launch_bf16; fi
    s=$(stale_h "$OUT/frontier_bf16.log"); [ "$s" -gt 5 ] && log "WARN BF16 log stale ${s}h"
  fi
  if [ -f "$OUT/frontier_bf16_DONE" ] && [ ! -f "$OUT/GPQA_ROUND_DONE" ]; then
    if ! pgrep -f 'gpqa_launch_1019.sh$' >/dev/null 2>&1 && ! pgrep -f 'chain_gpqa_1019.sh [FG]$' >/dev/null 2>&1; then
      log "GPQA drivers gone -> relaunch launcher"; launch_gpqa
    fi
    s=$(stale_h "$OUT/gpqa_F.log"); [ "$s" -gt 5 ] && log "WARN GQ-F log stale ${s}h"
    s=$(stale_h "$OUT/gpqa_G.log"); [ "$s" -gt 5 ] && log "WARN GQ-G log stale ${s}h"
  fi
  if [ -f "$OUT/GPQA_ROUND_DONE" ]; then log "ALL DONE - watchdog exits"; exit 0; fi
  sleep 600
done
log "watchdog max iterations (48h) reached"
