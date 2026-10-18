#!/bin/bash
# BF16 floor for the corrected non-think var_track (v3b).
# Context: 32K indicator (64K/60K infeasible: TP=32 KV cap ~35K).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL=glm52-bf16-eval
LOG="$OUT/bf16_v3b.log"
log(){ echo "[bf16v3b $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }
log "=== stop zzw on all 4 nodes (if any) ==="
for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
  ssh_q yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c; done" >>"$LOG" 2>&1
done
sleep 15
log "=== BF16 deploy (maxlen 36864, mem 0.95, eager, seqs 32) ==="
MAX_MODEL_LEN=36864 MEM_UTIL=0.95 ENFORCE_EAGER=1 MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "DEPLOY FAILED rc=$rc"; exit 1; fi
log "=== BF16: v3b (non-think cap 1024, 32K indicator) ==="
python3 "$P2/ruler_hard_probe_v3b.py" --api "$API" --model "$MODEL" --out "$OUT/r3b_bf16_ruler3b.jsonl" --length 32000 --workers 4 >>"$LOG" 2>&1
log "v3b rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/R3B_BF16_DONE"
log "BF16 V3B FLOOR DONE"
