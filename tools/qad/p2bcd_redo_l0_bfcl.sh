#!/bin/bash
# Re-run l0's #3 (BFCL) with the SAME protocol as the other products (16 workers).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_redo_l0.log
log(){ echo "[redo $(date '+%F %T')] $*" | tee -a "$LOG"; }
log "=== l0: deploy (maxlen 135168, mem 0.85) ==="
bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni \
SERVED=p2bcd_l0 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47 \
  MAX_MODEL_LEN=135168 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== l0: #3 BFCL re-run (16 workers, max_tokens=512) ==="
python3 "$P2/bfcl_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_l0 \
  --out "$OUT/p2bcd_l0_bfcl.jsonl" >>"$LOG" 2>&1
log "#3 rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/p2bcd_l0_bfcl_redo_DONE"
log "REDO DONE"
