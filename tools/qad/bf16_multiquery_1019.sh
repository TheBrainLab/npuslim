#!/bin/bash
# BF16 64K/128K multiquery NIAH (4 nodes, TP=32), 36 cases.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL="glm52-bf16-eval"
LOG="$OUT/mq_bf16.log"
log(){ echo "[mqbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/mq_bf16_DONE" ]; then log "already done - skip"; exit 0; fi
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== BF16 deploy (maxlen 135168, eager=$EAGER) ==="
MAX_MODEL_LEN=135168 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
log "=== BF16: multiquery 64K/128K ==="
python3 "$P2/niah_multiquery_probe.py" --api "$API" --model "$MODEL" \
  --out "$OUT/mq_bf16.jsonl" --lengths 65536,131072 --per-cell 6 --workers 2 >>"$LOG" 2>&1
rc=$?
log "multiquery rc=$rc"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; exit 1; fi
touch "$OUT/mq_bf16_DONE"
log "=== BF16 multiquery DONE ==="
