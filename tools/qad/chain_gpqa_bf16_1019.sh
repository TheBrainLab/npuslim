#!/bin/bash
# =============================================================================
# chain_gpqa_bf16_1019.sh — BF16 GPQA (4-node TP=32), appended AFTER the
# products' GPQA round (needs all 4 nodes). Pool at TP=32 is large (~300K+
# tokens at maxlen 36864) so batch 8 is safe; batch is set in the cfg template.
# Idempotent: skips when gpqa_bf16_DONE exists.
# =============================================================================
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
LOG="$OUT/gpqa_bf16.log"
log(){ echo "[gpqa-bf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }

if [ -f "$OUT/gpqa_bf16_DONE" ]; then log "already done - skip"; exit 0; fi
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

log "=== BF16 GPQA (official recipe, full 198) ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no yult@10.1.0.106 \
  "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh bf16 9081 bf16 gpqa_gen_0_shot_cot_chat_prompt" >>"$LOG" 2>&1
log "gpqa rc=$?"

bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/gpqa_bf16_DONE"
log "BF16 GPQA DONE"
