#!/bin/bash
# =============================================================================
# chain_tier1b_1016.sh — 1014 Tier1 剩余轴：Global-MMLU-Lite（低资源尾部）+
# SimpleQA（弃答/校准）+ IFBench（可验证约束）。
#   per product: deploy (maxlen 8192) -> gmmlu -> simpleqa -> ifbench -> stop
# 用法: LANE=<tag> HEAD_IP=.. WORKER_IP=.. PRODUCTS="tag|model|mem ..." bash chain_tier1b_1016.sh
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-T1B}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
CONTAINER=e6_noawq_env
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/tier1b_${LANE}.log"
log(){ echo "[t1b$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }

run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/t1b_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 8192, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST \
    bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="t1b_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=8192 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip probes"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: Global-MMLU-Lite (bn/sw/yo/my + en) ==="
  python3 "$P2/gmmlu_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1b_${tag}" \
    --out "$OUT/t1b_${tag}_gmmlu.jsonl" >>"$LOG" 2>&1
  log "gmmlu rc=$?"
  log "=== $tag: SimpleQA ==="
  python3 "$P2/simpleqa_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1b_${tag}" \
    --out "$OUT/t1b_${tag}_simpleqa.jsonl" >>"$LOG" 2>&1
  log "simpleqa rc=$?"
  log "=== $tag: IFBench ==="
  python3 "$P2/ifbench_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1b_${tag}" \
    --out "$OUT/t1b_${tag}_ifbench.jsonl" >>"$LOG" 2>&1
  log "ifbench rc=$?"
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/t1b_${tag}_DONE"
  log "=== $tag DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/T1B_${LANE}_DONE"
log "TIER1B LANE $LANE DONE"
