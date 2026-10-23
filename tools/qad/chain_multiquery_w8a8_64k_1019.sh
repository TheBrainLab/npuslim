#!/bin/bash
# w8a8 64K-only multiquery fallback (TP=16 cannot fit 135K).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
HEAD_IP=10.1.0.106; WORKER_IP=10.1.0.135
MODEL="/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert"
LOG="$OUT/mq_G_64k.log"
log(){ echo "[mqw8a8-64k $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/mq_w8a8_DONE" ]; then log "already done - skip"; touch "$OUT/MQ_G_DONE"; exit 0; fi
log "=== w8a8 deploy (maxlen 69632, mem 0.95) ==="
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=yult@$HEAD_IP WORKER_HOST=yult@$WORKER_IP bash "$DEPLOY" --stop >>"$LOG" 2>&1 || true
sleep 8
MODEL="$MODEL" SERVED="mq_w8a8_64k" ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=yult@$HEAD_IP WORKER_HOST=yult@$WORKER_IP \
  MAX_MODEL_LEN=69632 MEM_UTIL=0.95 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=yult@$HEAD_IP WORKER_HOST=yult@$WORKER_IP bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
API="http://${HEAD_IP}:8000/v1/chat/completions"
rm -f "$OUT/mq_w8a8.jsonl"
log "=== w8a8 multiquery 64K only (18 samples) ==="
python3 "$P2/niah_multiquery_probe.py" --api "$API" --model "mq_w8a8_64k" \
  --out "$OUT/mq_w8a8.jsonl" --lengths 65536 --per-cell 6 --workers 2 >>"$LOG" 2>&1
rc=$?
log "multiquery rc=$rc"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=yult@$HEAD_IP WORKER_HOST=yult@$WORKER_IP bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; exit 1; fi
touch "$OUT/mq_w8a8_DONE"
touch "$OUT/MQ_G_DONE"
log "=== w8a8 64K multiquery DONE; MQ_G_DONE touched ==="
