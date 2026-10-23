#!/bin/bash
# HLE-MC P2-long calibration on l0: thinking high, cap 16384, workers 4.
# Same 10-question smoke set as P2-8K for a controlled comparison.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="${DATA:-$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_smoke10.parquet}"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
HEAD_IP="${HEAD_IP:-10.1.0.106}"; WORKER_IP="${WORKER_IP:-10.1.0.135}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
MODEL="${MODEL:-/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni}"
SERVED="${SERVED:-hle_mc_p2long}"
LOG="$OUT/hle_mc_p2long.log"
log(){ echo "[hlep2long $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/hle_mc_p2long_DONE" ]; then log "already done - skip"; exit 0; fi
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1 || true
sleep 8
log "=== deploy l0 (maxlen 32768) on $HEAD_IP + $WORKER_IP ==="
MODEL="$MODEL" SERVED="$SERVED" ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
  MAX_MODEL_LEN=32768 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
API="http://${HEAD_IP}:8000/v1/chat/completions"
rm -f "$OUT/hle_mc_calib_p2long_l0.jsonl"
log "=== P2-long thinking high cap16384 workers4 ==="
python3 "$P2/hle_mc_probe.py" --api "$API" --model "$SERVED" --data "$DATA" \
  --out "$OUT/hle_mc_calib_p2long_l0.jsonl" --workers 4 --max-tokens 16384 \
  --enable-thinking --reasoning-effort high --temperature 0.6 >>"$LOG" 2>&1
rc=$?
log "p2long rc=$rc"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; exit 1; fi
touch "$OUT/hle_mc_p2long_DONE"
log "=== P2-long calibration DONE ==="
