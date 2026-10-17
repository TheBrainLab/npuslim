#!/bin/bash
# w8a8 #5 (NIAH 64K/128K) at TP=32 on ALL 4 nodes.
# Reason: 719GB at TP=16 cannot serve 128K (KV 12GiB > available); TP=32 halves the weights.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_4node_eval.sh
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
LOG=$OUT/p2bcd_w8a8_niah.log
log(){ echo "[w8a8niah $(date '+%F %T')] $*" | tee -a "$LOG"; }
log "=== w8a8: deploy TP=32 (4 nodes, maxlen 135168, mem 0.92) ==="
MODEL=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert \
SERVED=p2bcd_w8a8 PORT=9081 MAX_MODEL_LEN=135168 MEM_UTIL=0.92 \
MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32 ENFORCE_EAGER=0 \
  bash "$DEPLOY" >>"$LOG" 2>&1 || { log "deploy FAILED"; exit 1; }
log "=== w8a8: #5 NIAH 64K/128K ==="
python3 "$P2/niah_probe.py" --api "http://10.1.0.106:9081/v1/chat/completions" --model p2bcd_w8a8 --out "$OUT/p2bcd_w8a8_niah.jsonl" >>"$LOG" 2>&1
log "#5 rc=$?"
# cleanup on all 4 nodes
for ip in 10.1.0.106 10.1.0.57 10.1.0.47 10.1.0.135; do
  timeout 60 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new yult@$ip "docker exec e6_noawq_env bash -lc 'pkill -9 -f \"[v]llm serve\"; ray stop --force >/dev/null 2>&1; pkill -9 -f \"[r]ay::\"'" >/dev/null 2>&1 || true
done
touch "$OUT/p2bcd_w8a8_niah_DONE"
log "W8A8 NIAH LANE DONE"
