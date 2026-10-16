#!/bin/bash
# =============================================================================
# chain_p2a_only.sh — P2a (long-context mbpp) only, no router phase.
#   Per product: deploy (NO eager, no router hook) -> lm_eval mbpp_longctx -> stop.
#   Idempotent: skips a product whose lctx_<tag>/ already exists.
#
# env: LANE HEAD_IP WORKER_IP HEAD_HOST WORKER_HOST PRODUCTS("tag|path|mem ...")
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-P}}"
HEAD_IP="${HEAD_IP:-10.1.0.57}"; WORKER_IP="${WORKER_IP:-10.1.0.47}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
TASK_DIR="${TASK_DIR:-/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx}"
OUT_TAG_PREFIX="${OUT_TAG_PREFIX:-lctx}"
CONTAINER=e6_noawq_env
PORT=8000
MAXLEN=36864
CONC=6
LOG="$OUT/p2a_${LANE}.log"
log(){ echo "[p2a$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_long(){ timeout 7200 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if [ -d "$OUT/${OUT_TAG_PREFIX}_${tag}" ]; then log "$tag: P2a already done - skip"; return 0; fi
  log "=== $tag: deploy (no-eager, no router hook) ==="
  bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="p2a_${LANE}_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=$MAXLEN MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; return 1; fi
  log "=== $tag: metric preflight ==="
  if HEAD_HOST="$HEAD_HOST" CONTAINER="$CONTAINER" TASK_DIR="$TASK_DIR" \
       bash "$HOSTDIR/ylt/npuslim_repo/tools/qad/preflight_metric_1016.sh" >>"$LOG" 2>&1; then
    log "preflight OK (metric self-test = 1.0)"
  else
    log "$tag PREFLIGHT FAILED (metric broken; see log) - aborting product"
    bash "$DEPLOY" --stop >>"$LOG" 2>&1
    return 1
  fi
  log "=== $tag: lm_eval (max_gen_toks=160, conc=$CONC) ==="
  ssh_long "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $model $PORT $CONC $MAXLEN /llm_infer_workspace/ylt/glm52-qad/audit1009/${OUT_TAG_PREFIX}_${tag}" >>"$LOG" 2>&1
  log "lm_eval rc=$?"
  bash "$DEPLOY" --stop >>"$LOG" 2>&1
  log "=== $tag DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/P2A_${LANE}_DONE"
log "P2A LANE $LANE DONE"
