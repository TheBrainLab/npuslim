#!/bin/bash
# w8a8 probes on the G pair (.106+.135) in parallel with the main chain.
# NOTE: 719GB at TP=16 -> KV cannot serve 128K; maxlen capped at 69632 (64K NIAH only).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_w8a8_lane.log
log(){ echo "[w8a8lane $(date '+%F %T')] $*" | tee -a "$LOG"; }
log "=== w8a8: deploy (maxlen 69632, mem 0.92, G pair) ==="
HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
MODEL=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert \
SERVED=p2bcd_w8a8 ROUTER_DUMP="" ENFORCE_EAGER=0 \
  HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
  MAX_MODEL_LEN=69632 MEM_UTIL=0.92 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== w8a8: #3 BFCL ==="
python3 "$P2/bfcl_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model p2bcd_w8a8 --out "$OUT/p2bcd_w8a8_bfcl.jsonl" >>"$LOG" 2>&1
log "#3 rc=$?"
log "=== w8a8: #4 safety ==="
python3 "$P2/safety_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model p2bcd_w8a8 --out "$OUT/p2bcd_w8a8_safety.jsonl" >>"$LOG" 2>&1
log "#4 rc=$?"
log "=== w8a8: #5 NIAH (64K only) ==="
python3 "$P2/niah_probe.py" --api "http://10.1.0.106:8000/v1/chat/completions" --model p2bcd_w8a8 --lengths 65536 --out "$OUT/p2bcd_w8a8_niah.jsonl" >>"$LOG" 2>&1
log "#5 rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/p2bcd_w8a8_DONE"
log "W8A8 LANE DONE"
