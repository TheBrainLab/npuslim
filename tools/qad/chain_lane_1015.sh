#!/bin/bash
# =============================================================================
# chain_lane_1015.sh — parameterized P1 (router) + P2a (long-context mbpp) lane.
#   One lane = one TP=16 pair (HEAD+WORKER), running a list of products
#   sequentially: deploy (maxlen=36864 + routed-experts) -> router probe ->
#   lm_eval mbpp_longctx -> stop.
#
# env:
#   LANE       lane tag (for the done marker / logs)
#   HEAD_IP WORKER_IP HEAD_HOST WORKER_HOST   node pair (default .57 + .47)
#   PRODUCTS   "tag|container_path" entries, space separated
#
# Usage:
#   LANE=A HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 PRODUCTS="l0|/path w4a8c8|/path" \
#     setsid nohup bash tools/qad/chain_lane_1015.sh >/tmp/lane_A.log 2>&1 &
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-A}}"
HEAD_IP="${HEAD_IP:-10.1.0.57}"; WORKER_IP="${WORKER_IP:-10.1.0.47}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PRODUCTS="${PRODUCTS:?need PRODUCTS}"

HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
QAD_C=/llm_infer_workspace/ylt/npuslim_repo/tools/qad
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
RDIR="$OUT/router_dump"
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx
CONTAINER=e6_noawq_env
PORT=8000
MAXLEN=36864
CONC=4
LOG="$OUT/lane_${LANE}.log"
mkdir -p "$RDIR"
log(){ echo "[lane$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_long(){ timeout 7200 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

export HEAD_IP WORKER_IP HEAD_HOST WORKER_HOST

run_product(){
  local tag="$1" model="$2" memutil="${3:-0.85}"
  log "=== $tag: stop ==="; bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 10
  log "=== $tag: deploy (maxlen=$MAXLEN, mem=$memutil, eager+router-dump) on $HEAD_IP+$WORKER_IP ==="
  MODEL="$model" SERVED="p1_${LANE}_${tag}" \
    ROUTER_DUMP="/llm_infer_workspace/ylt/glm52-qad/audit1009/router_dump/router_${tag}.pt" \
    ENFORCE_EAGER=1 \
    ROUTER_CALLS="${ROUTER_CALLS:-4}" \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=$MAXLEN MEM_UTIL="$memutil" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  [ $? -ne 0 ] && { log "$tag DEPLOY FAILED"; return 1; }
  log "=== $tag: P1 router probe (patch dump) ==="
  ssh_q "$HEAD_HOST" "docker exec $CONTAINER python3 $QAD_C/router_probe_patch.py --api http://127.0.0.1:${PORT}/v1/completions --model p1_${LANE}_${tag} --tokenizer $model --parquet /llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet --tokens 2048" >>"$LOG" 2>&1
  log "router probe rc=$?"
  log "=== $tag: P2a long-context mbpp ==="
  ssh_long "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $model $PORT $CONC $MAXLEN /llm_infer_workspace/ylt/glm52-qad/audit1009/lctx_${tag}" >>"$LOG" 2>&1
  log "lm_eval rc=$?"
  log "=== $tag DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done

log "lane done - stopping serve"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/LANE_${LANE}_DONE"
log "LANE $LANE ALL DONE"
