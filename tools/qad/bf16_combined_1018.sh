#!/bin/bash
# BF16 combined audit round: ifbench@1024 + simpleqa@1024 + v3b27@32K (one 4-node deploy).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL=glm52-bf16-eval
LOG="$OUT/bf16c.log"
log(){ echo "[bf16c $(date '+%F %T')] $*" | tee -a "$LOG"; }
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
log "=== BF16: ifbench (cap 1024) ==="
python3 "$P2/ifbench_probe_c1024.py" --api "$API" --model "$MODEL" --out "$OUT/bf16c_ifbench.jsonl" --workers 32 >>"$LOG" 2>&1
log "ifbench rc=$?"
log "=== BF16: simpleqa (cap 1024) ==="
python3 "$P2/simpleqa_probe_c1024.py" --api "$API" --model "$MODEL" --out "$OUT/bf16c_simpleqa.jsonl" --workers 32 >>"$LOG" 2>&1
log "simpleqa rc=$?"
log "=== BF16: v3b27 (32K indicator) ==="
python3 "$P2/ruler_hard_probe_v3b27.py" --api "$API" --model "$MODEL" --out "$OUT/bf16c_v3b27.jsonl" --length 32000 --workers 4 >>"$LOG" 2>&1
log "v3b27 rc=$?"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/bf16c_DONE"
log "BF16 COMBINED DONE"
