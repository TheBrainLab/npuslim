#!/bin/bash
# BF16 arm for the 1019 frontier benchmarks (OneRuler + AIME), 4-node TP=32.
# Uses cudagraph if the test passed (bf16_cudagraph_OK), else eager.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
AB="$HOSTDIR/ylt/glm52-qad/aisbench"
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL=glm52-bf16-eval
LOG="$OUT/frontier_bf16.log"
log(){ echo "[frontbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== stop zzw on all 4 nodes (if any) ==="
for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
  ssh_q yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c; done" >>"$LOG" 2>&1
done
sleep 15
log "=== BF16 deploy (36864/0.95, eager=$EAGER, seqs 32) ==="
MAX_MODEL_LEN=36864 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "DEPLOY FAILED rc=$rc"; exit 1; fi
log "=== BF16: OneRuler ==="
python3 "$P2/oneruler_probe.py" --api "$API" --model "$MODEL" --out "$OUT/frontier_bf16_oneruler.jsonl" --workers 4 >>"$LOG" 2>&1
log "oneruler rc=$?"
log "=== BF16: AISBench AIME2026 (container-side, port 9081) ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no yult@10.1.0.106 \
  "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh bf16 9081 bf16" >>"$LOG" 2>&1
log "aisbench rc=$?"
mkdir -p "$OUT/frontier_bf16_aisbench"
exp=$(ls -td "$AB"/outputs_bf16/*/ 2>/dev/null | head -1)
cp -r "$exp"results/*/ "$OUT/frontier_bf16_aisbench/" 2>/dev/null || true
cp "$exp"summary/*.csv "$exp"summary/*.md "$OUT/frontier_bf16_aisbench/" 2>/dev/null || true
cp "$exp"predictions/vllm-api-general-chat/*.jsonl "$OUT/frontier_bf16_aisbench/" 2>/dev/null || true
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/frontier_bf16_DONE"
log "BF16 FRONTIER DONE"
