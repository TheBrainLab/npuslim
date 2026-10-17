#!/bin/bash
# w8a8 long-form panel re-run (max_tokens was 1024 -> 100% truncation).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/t1c_w8a8_longform2.log
log(){ echo "[w8lf2 $(date '+%F %T')] $*" | tee -a "$LOG"; }
HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135
log "=== w8a8: deploy (G pair, maxlen 8192, mem 0.92) ==="
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert \
SERVED=t1c_w8a8 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST \
  MAX_MODEL_LEN=8192 MEM_UTIL=0.92 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== w8a8: long-form panel (re-run, 2048 cap) ==="
python3 "$P2/longform_panel_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model t1c_w8a8 --out "$OUT/t1c_w8a8_longform.jsonl" >>"$LOG" 2>&1
log "longform rc=$?"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/t1c_w8a8_longform2_DONE"
log "W8A8 LONGFORM REDO DONE"
