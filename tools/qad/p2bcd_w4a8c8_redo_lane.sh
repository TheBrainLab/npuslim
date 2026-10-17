#!/bin/bash
# w4a8c8 probes redo on the F pair (its BCD3 run was killed by a stray --stop).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_w4a8c8_redo.log
log(){ echo "[w4a8c8redo $(date '+%F %T')] $*" | tee -a "$LOG"; }
HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47
log "=== w4a8c8: deploy (F pair, maxlen 135168, mem 0.85) ==="
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/models/GLM-5.2-w4a8c8 SERVED=p2bcd_w4a8c8 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST \
  MAX_MODEL_LEN=135168 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== w4a8c8: #3 BFCL ==="
python3 "$P2/bfcl_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_w4a8c8 --out "$OUT/p2bcd_w4a8c8_bfcl.jsonl" >>"$LOG" 2>&1
log "#3 rc=$?"
log "=== w4a8c8: #4 safety ==="
python3 "$P2/safety_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_w4a8c8 --out "$OUT/p2bcd_w4a8c8_safety.jsonl" >>"$LOG" 2>&1
log "#4 rc=$?"
log "=== w4a8c8: #5 NIAH 64K/128K ==="
python3 "$P2/niah_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_w4a8c8 --out "$OUT/p2bcd_w4a8c8_niah.jsonl" >>"$LOG" 2>&1
log "#5 rc=$?"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/p2bcd_w4a8c8_redo_DONE"
log "W4A8C8 REDO DONE"
