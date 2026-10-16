#!/bin/bash
# =============================================================================
# chain_p1p2a_1015.sh — P1 (router consistency) + P2a (long-context mbpp)
#   For each product: deploy once (maxlen=36864, routed-experts on) ->
#   router probe (P1) -> lm_eval mbpp_longctx (P2a) -> stop.
#
# Usage: setsid nohup bash tools/qad/chain_p1p2a_1015.sh > /tmp/chain_p1p2a.log 2>&1 &
# =============================================================================
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
QAD_C=/llm_infer_workspace/ylt/npuslim_repo/tools/qad
HEAD_HOST="${HEAD_HOST:-yult@10.1.0.57}"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
RDIR="$OUT/router_dump"
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx
CONTAINER=e6_noawq_env
PORT=8000
MAXLEN=36864
CONC=4
LOG="$OUT/chain_p1p2a.log"
mkdir -p "$RDIR"
log(){ echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
dexec(){ ssh_q "$HEAD_HOST" "docker exec $CONTAINER $*"; }

W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
run_product(){
  local tag=$1 model=$2
  log "=== $tag: stop ==="; bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 10
  log "=== $tag: deploy (maxlen=$MAXLEN, routed-experts) ==="
  MODEL="$model" SERVED="p1_$tag" ROUTED_EXPERTS=1 \
    MAX_MODEL_LEN=$MAXLEN MEM_UTIL=0.85 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  [ $? -ne 0 ] && { log "$tag DEPLOY FAILED"; return 1; }
  log "=== $tag: P1 router probe ==="
  ssh_q "$HEAD_HOST" "docker exec $CONTAINER python3 $QAD_C/router_probe.py --api http://127.0.0.1:${PORT}/v1/completions --model p1_$tag --tokenizer $model --parquet /llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet --tokens 2048 --out /llm_infer_workspace/ylt/glm52-qad/audit1009/router_dump/router_${tag}.npz" >>"$LOG" 2>&1
  log "router probe rc=$?"
  log "=== $tag: P2a long-context mbpp ==="
  ssh_q "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $model $PORT $CONC $MAXLEN /llm_infer_workspace/ylt/glm52-qad/audit1009/lctx_${tag}" >>"$LOG" 2>&1
  log "lm_eval rc=$?"
  log "=== $tag DONE ==="
}

run_product l0     "$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni"
run_product p2clip "$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234"
run_product w4a8c8 "/llm_infer_workspace/models/GLM-5.2-w4a8c8"

log "stopping final serve"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/P1P2A_DONE"
log "ALL DONE"
