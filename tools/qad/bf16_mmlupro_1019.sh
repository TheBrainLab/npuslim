#!/bin/bash
# BF16 arm for MMLU-Pro 500 (4 nodes, TP=32).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DATA="$HOSTDIR/ylt/glm52-qad/p2bcd_data/frontier/mmlu_pro/mmlu_pro_500.parquet"
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL="glm52-bf16-eval"
LOG="$OUT/mmlupro_bf16.log"
log(){ echo "[mmpbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$OUT/mmlupro_bf16_DONE" ]; then log "already done - skip"; exit 0; fi
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== BF16 deploy (maxlen 16384, eager=$EAGER) ==="
MAX_MODEL_LEN=16384 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=4096 \
  bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
log "=== BF16: MMLU-Pro 500 ==="
python3 "$P2/mmlu_pro_probe.py" --api "$API" --model "$MODEL" --data "$DATA" \
  --out "$OUT/mmlupro_bf16.jsonl" --workers 16 >>"$LOG" 2>&1
rc=$?
log "mmlupro rc=$rc"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc (not marking DONE)"; exit 1; fi
touch "$OUT/mmlupro_bf16_DONE"
log "=== BF16 MMLU-Pro DONE ==="
