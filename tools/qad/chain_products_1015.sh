#!/bin/bash
# =============================================================================
# chain_products_1015.sh
#   Unattended chain: after l0 (already done), sequentially deploy each remaining
#   W4A8 product and run the L0 instruments on it, so the delivery matrix
#   (l0 / p2clip_s1234 / w4a8c8) is produced without manual intervention.
#
#   Order: p2clip_s1234 -> w4a8c8 (production target)
#   Marker: $OUT/CHAIN_DONE on success; log: $OUT/chain_products.log
# =============================================================================
set -uo pipefail
BASEH=/home/yult/llm_infer_workspace
DEPLOY=$BASEH/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
RUNNER=$BASEH/ylt/npuslim_repo/tools/qad/run_l0_instruments_1015.sh
OUT=$BASEH/ylt/glm52-qad/audit1009/l0_fidelity_1015
LOG=$OUT/chain_products.log
mkdir -p "$OUT"

log(){ echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_json(){ local f=$1; for i in $(seq 1 600); do [ -f "$f" ] && return 0; sleep 30; done; return 1; }

run_product(){
  local tag=$1 model=$2 served=$3
  log "=== $tag: stop current serve ==="
  bash "$DEPLOY" --stop >>"$LOG" 2>&1
  sleep 12
  log "=== $tag: deploy ${model} ==="
  MODEL="$model" SERVED="$served" MEM_UTIL=0.88 MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED - skipping"; return 1; fi
  log "=== $tag: launch instruments ==="
  TAG="$tag" MODEL_NAME="$served" ARTIFACT="$model" bash "$RUNNER" >>"$LOG" 2>&1
  wait_json "$OUT/dist_fidelity_${tag}.json"    || log "$tag fidelity timeout"
  wait_json "$OUT/ppl_length_val_${tag}.json"   || log "$tag ppl-val timeout"
  wait_json "$OUT/ppl_length_train_${tag}.json" || log "$tag ppl-train timeout"
  log "=== $tag DONE ==="
  return 0
}

log "chain start (l0 already complete)"
run_product p2clip /llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234 glm52-w4a8-p2clip
run_product w4a8c8 /llm_infer_workspace/models/GLM-5.2-w4a8c8 glm52-w4a8c8
log "stopping final serve"
bash "$DEPLOY" --stop >>"$LOG" 2>&1
log "ALL PRODUCTS DONE"
touch "$OUT/CHAIN_DONE"
