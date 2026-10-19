#!/bin/bash
# 1019 frontier benchmarks (P1/P2/P3): OneRuler (multilingual NIAH) + AIME 2026 + GPQA-Diamond.
# Per product: deploy (maxlen 69632) -> oneruler -> aime (official recipe) -> stop.
# (GPQA moved to the dedicated round: chain_gpqa_1019.sh, batch tuned by KV pool.)
set -uo pipefail
LANE="${1:-${LANE:-FR}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
AB="$HOSTDIR/ylt/glm52-qad/aisbench"
ABCFG="$HOSTDIR/ylt/glm52-qad/aisbench_cfgs"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/frontier_${LANE}.log"
log(){ echo "[front$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/frontier_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 69632, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="front_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=69632 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  log "=== $tag: OneRuler (multilingual, sampled) ==="
  python3 "$P2/oneruler_probe.py" --api "$api" --model "front_${tag}" --out "$OUT/frontier_${tag}_oneruler.jsonl" --workers 4 >>"$LOG" 2>&1
  log "oneruler rc=$?"
  log "=== $tag: AISBench AIME2026 (official recipe, container-side; GPQA in dedicated round) ==="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD_HOST" \
    "docker exec e6_noawq_env bash /llm_infer_workspace/ylt/glm52-qad/aisbench_run.sh ${tag} 8000 ${tag}" >>"$LOG" 2>&1
  log "aisbench rc=$?"
  local exp
  exp=$(ls -td "$AB"/outputs_${tag}/*/ 2>/dev/null | head -1)
  mkdir -p "$OUT/frontier_${tag}_aisbench"
  cp -r "$exp"results/*/ "$OUT/frontier_${tag}_aisbench/" 2>/dev/null || true
  cp "$exp"summary/*.csv "$exp"summary/*.md "$OUT/frontier_${tag}_aisbench/" 2>/dev/null || true
  cp "$exp"predictions/vllm-api-general-chat/*.jsonl "$OUT/frontier_${tag}_aisbench/" 2>/dev/null || true
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/frontier_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/FRONTIER_${LANE}_DONE"
log "FRONTIER LANE $LANE DONE"
