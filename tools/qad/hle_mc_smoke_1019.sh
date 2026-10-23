#!/bin/bash
# HLE multiple-choice 10-question smoke on l0 (2 nodes), thinking high.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_smoke10.parquet"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
HEAD_IP="${HEAD_IP:-10.1.0.106}"; WORKER_IP="${WORKER_IP:-10.1.0.135}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
MODEL="${MODEL:-/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni}"
SERVED="${SERVED:-hle_mc_l0}"
LOG="$OUT/hle_mc_smoke.log"
log(){ echo "[hlemc $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/hle_mc_smoke_DONE" ]; then log "already done - skip"; exit 0; fi
log "=== deploy l0 (maxlen 32768, mem 0.85) on $HEAD_IP + $WORKER_IP ==="
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1 || true
sleep 8
MODEL="$MODEL" SERVED="$SERVED" ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
  MAX_MODEL_LEN=32768 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
API="http://${HEAD_IP}:8000/v1/chat/completions"
log "=== HLE-MC smoke10 ==="
python3 "$P2/hle_mc_probe.py" --api "$API" --model "$SERVED" --data "$DATA" \
  --out "$OUT/hle_mc_smoke10_l0.jsonl" --workers 4 --max-tokens 16384 >>"$LOG" 2>&1
rc=$?
log "hle-mc rc=$rc"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; exit 1; fi
touch "$OUT/hle_mc_smoke_DONE"
log "=== HLE-MC smoke DONE ==="
