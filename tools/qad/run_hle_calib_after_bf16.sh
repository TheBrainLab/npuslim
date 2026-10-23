#!/bin/bash
# Wait for BF16 MMLU-Pro DONE, then run HLE-MC protocol calibration.
# Calibration only selects P0/P1/P2; it does NOT start the 60x3 pilot.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
LOG="$OUT/hle_calib_after_bf16.log"
log(){ echo "[hlewait $(date '+%F %T')] $*" | tee -a "$LOG"; }

BF16_PID=$(cat "$OUT/mmlupro_bf16.pid" 2>/dev/null || true)
deadline=$(( $(date +%s) + 5*3600 ))
while [ ! -f "$OUT/mmlupro_bf16_DONE" ]; do
  if [ -n "$BF16_PID" ] && ! kill -0 "$BF16_PID" 2>/dev/null; then
    log "BF16 process $BF16_PID died without DONE; not starting calibration"
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "BF16 timeout; not starting calibration"
    exit 1
  fi
  sleep 60
done
log "BF16 MMLU-Pro DONE; starting HLE-MC calibration (P0->P1->P2, l0, 10 questions)"
setsid nohup bash "$QAD/hle_mc_calib_1019.sh" >>"$LOG" 2>&1 < /dev/null &
CAL_PID=$!
echo "$CAL_PID" > "$OUT/hle_mc_calib.pid"
deadline=$(( $(date +%s) + 3*3600 ))
while [ ! -f "$OUT/hle_mc_calib_DONE" ]; do
  if ! kill -0 "$CAL_PID" 2>/dev/null; then
    log "calibration process $CAL_PID died without DONE"
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "calibration timeout"
    exit 1
  fi
  sleep 60
done
log "HLE-MC calibration DONE; see hle_mc_calib.log + hle_mc_calib_p{0,1,2}_l0.jsonl"
