#!/bin/bash
# BF16 32K-only multiquery floor (4 nodes, TP=32), 18 cases.
# This is NOT comparable to the 64K/128K multiquery table; it exists only as a
# BF16 floor at a max_model_len that fits the 4-node KV budget.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL="glm52-bf16-eval"
LOG="$OUT/mq_bf16_32k.log"
log(){ echo "[mqbf16-32k $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/mq_bf16_32k_DONE" ]; then log "already done - skip"; exit 0; fi
rm -f "$OUT/mq_bf16_32k_FAILED"
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== BF16 deploy 32K (maxlen 32768, eager=$EAGER) ==="
MAX_MODEL_LEN=32768 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 \
  bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "DEPLOY FAILED rc=$rc"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; touch "$OUT/mq_bf16_32k_FAILED"; exit 1; fi
log "=== BF16 32K multiquery (18 cases) ==="
python3 "$P2/niah_multiquery_probe.py" --api "$API" --model "$MODEL" \
  --out "$OUT/mq_bf16_32k.jsonl" --lengths 32768 --per-cell 6 --workers 2 >>"$LOG" 2>&1
rc=$?
log "multiquery 32K rc=$rc"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; touch "$OUT/mq_bf16_32k_FAILED"; exit 1; fi
touch "$OUT/mq_bf16_32k_DONE"
log "=== BF16 32K multiquery DONE ==="
