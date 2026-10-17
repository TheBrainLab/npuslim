#!/bin/bash
# w8a8 #3 (BFCL) + #4 (safety) on the G pair (TP=16, maxlen 8192).
# The #5 (NIAH 64K/128K) needs TP=32 (4 nodes) and runs later.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_w8a8_b34.log
log(){ echo "[w8a8b34 $(date '+%F %T')] $*" | tee -a "$LOG"; }
log "=== w8a8: deploy (G pair, maxlen 8192, mem 0.92) ==="
HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert \
SERVED=p2bcd_w8a8 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
  MAX_MODEL_LEN=8192 MEM_UTIL=0.92 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== w8a8: #3 BFCL ==="
python3 "$P2/bfcl_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model p2bcd_w8a8 --out "$OUT/p2bcd_w8a8_bfcl.jsonl" >>"$LOG" 2>&1
log "#3 rc=$?"
log "=== w8a8: #4 safety ==="
python3 "$P2/safety_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model p2bcd_w8a8 --out "$OUT/p2bcd_w8a8_safety.jsonl" >>"$LOG" 2>&1
log "#4 rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/p2bcd_w8a8_b34_DONE"
log "W8A8 B34 DONE"
