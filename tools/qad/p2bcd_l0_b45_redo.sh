#!/bin/bash
# l0: re-run #4 (safety) + #5 (NIAH) — the originals were damaged by a stray stop at 11:53:47.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_l0_b45.log
log(){ echo "[l0b45 $(date '+%F %T')] $*" | tee -a "$LOG"; }
HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47
log "=== l0: deploy (F pair, maxlen 135168, mem 0.85) ==="
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni \
SERVED=p2bcd_l0 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST \
  MAX_MODEL_LEN=135168 MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== l0: #4 safety (re-run) ==="
python3 "$P2/safety_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_l0 --out "$OUT/p2bcd_l0_safety.jsonl" >>"$LOG" 2>&1
log "#4 rc=$?"
log "=== l0: #5 NIAH (re-run) ==="
python3 "$P2/niah_probe.py" --api "http://10.1.0.57:8000/v1/chat/completions" --model p2bcd_l0 --out "$OUT/p2bcd_l0_niah.jsonl" >>"$LOG" 2>&1
log "#5 rc=$?"
HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/p2bcd_l0_b45_redo_DONE"
log "L0 B45 REDO DONE"
