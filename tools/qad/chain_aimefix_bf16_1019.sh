#!/bin/bash
# BF16 AIME rerun with reasoning_effort=high (4-node TP=32).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
AB=$HOSTDIR/ylt/glm52-qad/aisbench
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
LOG="$OUT/aimefix_bf16_re.log"
log(){ echo "[aimefixbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }
if [ -f "$OUT/aimefix_bf16_re_DONE" ]; then log "already done - skip"; exit 0; fi
if [ -f "$OUT/bf16_cudagraph_OK" ]; then EAGER=0; else EAGER=1; fi
log "=== stop zzw on all 4 nodes (if any) ==="
for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
  ssh_q yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c; done" >>"$LOG" 2>&1
done
sleep 15
log "=== BF16 deploy (maxlen 131072, mem 0.95, eager=$EAGER) ==="
MAX_MODEL_LEN=36864 MEM_UTIL=0.95 ENFORCE_EAGER=$EAGER MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "DEPLOY FAILED rc=$rc"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
log "=== BF16 AIME (high effort, 65536) ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no yult@10.1.0.106 \
  "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh bf16_re 9081 bf16_re aime2026_gen_0_shot_chat_prompt" >>"$LOG" 2>&1
rc=$?
log "aime rc=$rc"
if [ $rc -ne 0 ]; then log "RUN FAILED rc=$rc"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; exit 1; fi
exp=$(ls -td "$AB"/outputs_bf16_re/*/ 2>/dev/null | head -1)
mkdir -p "$OUT/aimefix_bf16_re_aisbench"
cp -r "$exp"results/*/ "$OUT/aimefix_bf16_re_aisbench/" 2>/dev/null || true
cp "$exp"summary/*.csv "$exp"summary/*.md "$OUT/aimefix_bf16_re_aisbench/" 2>/dev/null || true
cp "$exp"predictions/vllm-api-general-chat/*.jsonl "$OUT/aimefix_bf16_re_aisbench/" 2>/dev/null || true
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/aimefix_bf16_re_DONE"
log "BF16 AIME (high) DONE"
