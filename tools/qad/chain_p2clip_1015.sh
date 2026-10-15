#!/bin/bash
# Re-run ONLY p2clip_s1234 with a thorough cleanup (docker restart clears stale ray
# actors / placement groups, AGENTS #16), then run the L0 instruments.
set -uo pipefail
BASEH=/home/yult/llm_infer_workspace
DEPLOY=$BASEH/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh
RUNNER=$BASEH/ylt/npuslim_repo/tools/qad/run_l0_instruments_1015.sh
OUT=$BASEH/ylt/glm52-qad/audit1009/l0_fidelity_1015
LOG=$OUT/p2clip_only.log
MODEL=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234
SERVED=glm52-w4a8-p2clip
log(){ echo "[p2clip $(date '+%F %T')] $*" | tee -a "$LOG"; }

log "restart containers (clean ray/python)"
for h in 10.1.0.57 10.1.0.47; do
  timeout 90 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" "docker restart e6_noawq_env >/dev/null 2>&1 && echo restarted $h"
done
sleep 20

log "deploy p2clip (MEM_UTIL=0.88)"
MODEL="$MODEL" SERVED="$SERVED" MEM_UTIL=0.88 MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=2048 bash "$DEPLOY" >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "DEPLOY FAILED - abort"; exit 1; fi
log "deploy OK"

log "launch instruments"
TAG=p2clip MODEL_NAME="$SERVED" ARTIFACT="$MODEL" bash "$RUNNER" >>"$LOG" 2>&1

log "waiting for 3 JSONs"
for name in dist_fidelity ppl_length_val ppl_length_train; do
  for i in $(seq 1 600); do
    [ -f "$OUT/${name}_p2clip.json" ] && { log "$name OK"; break; }
    sleep 30
  done
done
log "DONE"
touch "$OUT/P2CLIP_DONE"
