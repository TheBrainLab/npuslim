#!/bin/bash
# BF16 efficiency reference (4 nodes TP=32), after the product supplement lanes.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
LOG="$OUT/supp_bf16.log"
log(){ echo "[suppbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }
log "=== stop zzw on all 4 nodes (if any) ==="
for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
  ssh_q yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c; done" >>"$LOG" 2>&1
done
sleep 15
log "=== BF16 deploy (36864/0.95/eager/32) ==="
MAX_MODEL_LEN=36864 MEM_UTIL=0.95 ENFORCE_EAGER=1 MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "DEPLOY FAILED rc=$rc"; exit 1; fi
log "=== BF16 efficiency ==="
python3 "$P2/efficiency_probe.py" --api "http://10.1.0.106:9081/v1/chat/completions" --model glm52-bf16-eval --out "$OUT/supp_bf16_efficiency.json" >>"$LOG" 2>&1
log "eff rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/supp_bf16_DONE"
log "BF16 EFF DONE"
