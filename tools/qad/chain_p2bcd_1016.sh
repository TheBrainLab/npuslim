#!/bin/bash
# =============================================================================
# chain_p2bcd_1016.sh — #3/#4/#5 decision-axis probes for the 4 products.
#   per product: deploy (maxlen 135168) -> #3 BFCL -> #4 safety -> #5 NIAH -> stop
#   #3 P2b: BFCL v4 irrelevance/refusal-to-call + name-level relevance
#   #4 P2c: OR-Bench over-refusal + StrongREJECT refusal
#   #5 P2d: local NIAH 64K/128K (RULER-style)
#   2 nodes (F pair .57+.47); probes run on the control host over the node IP.
# env: LANE (default BCD)
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-BCD}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP=10.1.0.57; WORKER_IP=10.1.0.47
HEAD_HOST=yult@10.1.0.57; WORKER_HOST=yult@10.1.0.47
CONTAINER=e6_noawq_env
PORT=8000
LOG="$OUT/p2bcd_${LANE}.log"
log(){ echo "[p2bcd$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
REPO=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq
PRODUCTS="w8a8|${REPO}/glm5-w8a8_replica_p2clipfold_w8expert|0.92 l0|${W4A8}/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 p2clip|${W4A8}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85"

run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -f "$OUT/p2bcd_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (maxlen 135168, mem $mem) ==="
  bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="p2bcd_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=135168 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip probes"; bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  log "=== $tag: #3 BFCL (irrelevance + relevance) ==="
  python3 "$P2/bfcl_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "p2bcd_${tag}" \
    --out "$OUT/p2bcd_${tag}_bfcl.jsonl" >>"$LOG" 2>&1
  log "#3 rc=$?"
  log "=== $tag: #4 safety (OR-Bench + StrongREJECT) ==="
  python3 "$P2/safety_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "p2bcd_${tag}" \
    --out "$OUT/p2bcd_${tag}_safety.jsonl" >>"$LOG" 2>&1
  log "#4 rc=$?"
  log "=== $tag: #5 NIAH 64K/128K ==="
  python3 "$P2/niah_probe.py" --api "http://${HEAD_IP}:${PORT}/v1/chat/completions" --model "p2bcd_${tag}" \
    --out "$OUT/p2bcd_${tag}_niah.jsonl" >>"$LOG" 2>&1
  log "#5 rc=$?"
  bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/p2bcd_${tag}_DONE"
  log "=== $tag DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/P2BCD_${LANE}_DONE"
log "P2BCD LANE $LANE DONE"
