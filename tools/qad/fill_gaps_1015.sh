#!/bin/bash
# =============================================================================
# fill_gaps_1015.sh — after both lanes finish, serially re-run ONLY the missing /
# low-quality artifacts, with all nodes otherwise idle (clean conditions).
#
#   For each product: needs_router (dump missing OR only the 1-token warmup call)
#   and/or needs_p2a (lctx_<tag>/ missing) -> deploy once and run only what is needed.
#
# Usage: setsid nohup bash tools/qad/fill_gaps_1015.sh >/tmp/fill_gaps.log 2>&1 &
# =============================================================================
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
QAD_C=/llm_infer_workspace/ylt/npuslim_repo/tools/qad
HEAD_IP=10.1.0.57; WORKER_IP=10.1.0.47
HEAD_HOST=yult@10.1.0.57; WORKER_HOST=yult@10.1.0.47
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
RDIR="$OUT/router_dump"
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx
CONTAINER=e6_noawq_env
PORT=8000
MAXLEN=36864
CONC=4
LOG="$OUT/fill_gaps.log"
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
log(){ echo "[gaps $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_long(){ timeout 7200 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

# wait for both lanes
log "waiting for both lanes to finish..."
for i in $(seq 1 240); do
  [ -f "$OUT/LANE_A_DONE" ] && [ -f "$OUT/LANE_B_DONE" ] && break
  sleep 60
done
log "lanes done (or timeout) - starting gap fill"

router_ok(){  # needs the dump to have a call with >1 token (not just the warmup)
  local f="$1"
  [ -f "$f" ] || return 1
  docker exec "$CONTAINER" python3 -c "
import torch,sys
d=torch.load('$f', map_location='cpu')
mx=max(t.shape[0] for calls in d['layers'].values() for t in calls)
sys.exit(0 if mx>1 else 1)
" 2>/dev/null
}

run_product(){
  local tag="$1" model="$2" mem="${3:-0.85}"
  local need_r=0 need_p=0
  router_ok "$RDIR/router_${tag}.pt" || need_r=1
  [ -d "$OUT/lctx_${tag}" ] || need_p=1
  if [ $need_r = 0 ] && [ $need_p = 0 ]; then log "$tag: nothing to do"; return 0; fi
  log "=== $tag: need_router=$need_r need_p2a=$need_p -> deploy ==="
  bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 10
  MODEL="$model" SERVED="p1_A_${tag}" \
    ROUTER_DUMP="/llm_infer_workspace/ylt/glm52-qad/audit1009/router_dump/router_${tag}.pt" \
    ENFORCE_EAGER=1 ROUTER_CALLS=4 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=$MAXLEN MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skip"; return 1; fi
  if [ $need_r = 1 ]; then
    log "--- $tag: router probe (4-call) ---"
    ssh_q "$HEAD_HOST" "docker exec $CONTAINER python3 $QAD_C/router_probe_patch.py --api http://127.0.0.1:${PORT}/v1/completions --model p1_A_${tag} --tokenizer $model --parquet /llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet --tokens 2048" >>"$LOG" 2>&1
    log "router rc=$?"
  fi
  if [ $need_p = 1 ]; then
    log "--- $tag: P2a long-context mbpp ---"
    ssh_long "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $model $PORT $CONC $MAXLEN /llm_infer_workspace/ylt/glm52-qad/audit1009/lctx_${tag}" >>"$LOG" 2>&1
    log "lm_eval rc=$?"
  fi
  log "=== $tag done ==="
}

run_product l0     "$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni" 0.85
run_product p2clip "$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234" 0.85
run_product w4a8c8  "/llm_infer_workspace/models/GLM-5.2-w4a8c8" 0.85
run_product w8a8    "/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert" 0.92

bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/GAPS_DONE"
log "ALL GAP FILL DONE"
