#!/bin/bash
# 1014 Tier1 剩余：④ RULER-hard（64K）+ ⑤ 长文损伤面板。
#   per product: deploy (maxlen 69632) -> ruler_hard -> longform -> stop
set -uo pipefail
LANE="${1:-${LANE:-T1C}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
CONTAINER=e6_noawq_env; PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LOG="$OUT/tier1c_${LANE}.log"
log(){ echo "[t1c$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/t1c_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 69632, mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="t1c_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=69632 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip probes"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: RULER-hard 64K ==="
  python3 "$P2/ruler_hard_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1c_${tag}" \
    --out "$OUT/t1c_${tag}_ruler.jsonl" >>"$LOG" 2>&1
  log "ruler rc=$?"
  log "=== $tag: long-form panel ==="
  python3 "$P2/longform_panel_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "t1c_${tag}" \
    --out "$OUT/t1c_${tag}_longform.jsonl" >>"$LOG" 2>&1
  log "longform rc=$?"
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/t1c_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/T1C_${LANE}_DONE"
log "TIER1C LANE $LANE DONE"
