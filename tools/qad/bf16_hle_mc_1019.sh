#!/bin/bash
# BF16 HLE-MC 60-question pilot (4 nodes, TP=32).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="${DATA:-$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/hle/hle_mc_60.parquet}"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL="glm52-bf16-eval"
PROTOCOL_ARGS="${PROTOCOL_ARGS:?}"
LOG="$OUT/hle_mc_bf16.log"
log(){ echo "[hlemcbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/hle_mc_bf16_DONE" ]; then log "already done - skip"; exit 0; fi
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== BF16 deploy (maxlen 32768, eager=$EAGER) ==="
MAX_MODEL_LEN=32768 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=4096 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
log "=== BF16: HLE-MC 60 ==="
python3 "$P2/hle_mc_probe.py" --api "$API" --model "$MODEL" --data "$DATA" \
  --out "$OUT/hle_mc_bf16.jsonl" $PROTOCOL_ARGS >>"$LOG" 2>&1
rc=$?
log "hle-mc rc=$rc"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; exit 1; fi
touch "$OUT/hle_mc_bf16_DONE"
log "=== BF16 HLE-MC DONE ==="
