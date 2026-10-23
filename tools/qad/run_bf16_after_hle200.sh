#!/bin/bash
# Complete the 5th multiquery arm (BF16) after HLE200 frees the 4 nodes.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
LOG="$OUT/bf16_late.log"
log(){ echo "[bf16late $(date '+%F %T')] $*" | tee -a "$LOG"; }
deadline=$(( $(date +%s) + 14*3600 ))
while [ ! -f "$OUT/HLE200_ALL_DONE" ]; do
  if [ "$(date +%s)" -gt "$deadline" ]; then log "HLE200 timeout; not starting BF16"; exit 1; fi
  sleep 120
done
if [ -f "$OUT/mq_bf16_DONE" ]; then log "BF16 already done - skip"; exit 0; fi
log "HLE200 DONE; starting late BF16 multiquery (64K/128K, 4 nodes)"
setsid nohup bash "$QAD/bf16_multiquery_1019.sh" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$OUT/bf16_late.pid"
log "BF16 late started pid=$!"
