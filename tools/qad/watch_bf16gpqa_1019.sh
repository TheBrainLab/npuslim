#!/bin/bash
# Watchdog v7: BF16 GPQA auto-relaunch.
set -uo pipefail
cd /home/yult/llm_infer_workspace/ylt
OUT=glm52-qad/audit1009
REPO=npuslim_repo/tools/qad
log(){ echo "[wbf16gpqa $(date '+%F %T')] $*" | tee -a "$OUT/watch_bf16gpqa.log"; }
log "start (pid $$)"
for it in $(seq 1 96); do
  if [ ! -f "$OUT/gpqa_bf16_DONE" ] && ! pgrep -f 'chain_gpqa_bf16_1019.sh$' >/dev/null 2>&1; then
    log "BF16 GPQA driver gone -> relaunch"
    setsid nohup bash "$REPO/chain_gpqa_bf16_1019.sh" >> "$OUT/gpqa_bf16.log" 2>&1 < /dev/null &
  fi
  [ -f "$OUT/gpqa_bf16_DONE" ] && { log "BF16 GPQA DONE - exit"; exit 0; }
  sleep 600
done
log "watchdog max iterations"
