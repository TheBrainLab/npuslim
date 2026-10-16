#!/bin/bash
# =============================================================================
# chain_final_1015.sh — final combined P1(router) + P2a(long-context mbpp) pass.
#
#   Per product (idempotent: skips what is already good):
#     A) if router dump is missing / only the 1-token warmup:
#          deploy WITH --enforce-eager (needed for the router hook) -> probe -> stop
#     B) if lctx_<tag>/ missing:
#          deploy WITHOUT eager (cudagraph -> ~3x faster decode) -> lm_eval -> stop
#
# env: LANE HEAD_IP WORKER_IP HEAD_HOST WORKER_HOST PRODUCTS("tag|path|mem ...")
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-F}}"
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
CONC=6
LOG="$OUT/final_${LANE}.log"
mkdir -p "$RDIR"
log(){ echo "[fin$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_long(){ timeout 7200 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

router_ok(){ local f="$1"; [ -f "$f" ] || return 1
  docker exec "$CONTAINER" python3 -c "
import torch,sys
d=torch.load('$f', map_location='cpu')
mx=max(t.shape[0] for v in d['layers'].values() for t in v)
sys.exit(0 if mx>1 else 1)" 2>/dev/null; }

deploy(){ local model="$1" served="$2" mem="$3" eager="$4" tag="$5"
  local rd=""
  [ "$eager" = "1" ] && rd="/llm_infer_workspace/ylt/glm52-qad/audit1009/router_dump/router_${tag}.pt"
  bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="$served" \
    ROUTER_DUMP="$rd" \
    ROUTER_CALLS=4 ENFORCE_EAGER="$eager" \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=$MAXLEN MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1; }

run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  if ! router_ok "$RDIR/router_${tag}.pt"; then
    log "=== $tag: router re-capture (eager) ==="
    if deploy "$model" "fin_${LANE}_${tag}_r" "$mem" 1 "$tag"; then
      ssh_q "$HEAD_HOST" "docker exec $CONTAINER python3 $QAD_C/router_probe_patch.py --api http://127.0.0.1:${PORT}/v1/completions --model fin_${LANE}_${tag}_r --tokenizer $model --parquet /llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet --tokens 2048" >>"$LOG" 2>&1
      log "router rc=$?"
    else
      log "$tag deploy(router) FAILED"
    fi
    bash "$DEPLOY" --stop >>"$LOG" 2>&1
  else
    log "$tag: router already good - skip"
  fi
  if [ ! -d "$OUT/lctx_${tag}" ]; then
    log "=== $tag: P2a long-context mbpp (no-eager) ==="
    deploy "$model" "fin_${LANE}_${tag}_p" "$mem" 0 "$tag" || { log "$tag deploy(p) FAILED"; return 1; }
    ssh_long "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $model $PORT $CONC $MAXLEN /llm_infer_workspace/ylt/glm52-qad/audit1009/lctx_${tag}" >>"$LOG" 2>&1
    log "lm_eval rc=$?"
    bash "$DEPLOY" --stop >>"$LOG" 2>&1
  else
    log "$tag: P2a already done - skip"
  fi
  log "=== $tag DONE ==="
}

for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; mem="${rest##*|}"
  [ "$mem" = "$model" ] && mem=0.85
  run_product "$tag" "$model" "$mem"
done
touch "$OUT/FINAL_${LANE}_DONE"
log "FINAL LANE $LANE DONE"
