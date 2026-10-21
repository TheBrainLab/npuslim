#!/bin/bash
# Night orchestration 10-20: after the current AIME runs release the lanes:
#   1) BF16_re (4 nodes, AIME high-effort)  -> aimefix_bf16_re_DONE
#   2) GPQA high-effort reruns (2 lanes): F: l0 -> w4a8c8 ; G: p2clip -> w8a8
# Idempotent via per-arm markers.
set -uo pipefail
cd /home/yult/llm_infer_workspace/ylt
OUT=glm52-qad/audit1009
REPO=npuslim_repo/tools/qad
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
W8=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq
LOG="$OUT/night_1019.log"
log(){ echo "[night $(date '+%F %T')] $*" | tee -a "$LOG"; }

# stage 1: wait for the current runs (l0_gh on F, p2clip_re on G)
for i in $(seq 1 48); do
  [ -f "$OUT/aimefix_l0_gh_DONE" ] && [ -f "$OUT/aimefix_p2clip_re_DONE" ] && break
  sleep 300
done
log "stage1: l0_gh=$([ -f "$OUT/aimefix_l0_gh_DONE" ] && echo ok || echo missing) p2clip_re=$([ -f "$OUT/aimefix_p2clip_re_DONE" ] && echo ok || echo missing)"
sleep 60

# stage 2: BF16 high-effort AIME (4 nodes)
if [ ! -f "$OUT/aimefix_bf16_re_DONE" ]; then
  log "stage2: BF16_re (4 nodes)"
  bash "$REPO/chain_aimefix_bf16_1019.sh" >> "$LOG" 2>&1
  log "stage2 rc=$?"
fi

# stage 3: GPQA high-effort reruns (2 lanes, high effort template _re)
log "stage3: GPQA re-runs (high effort)"
env LANE=F CFGSUF=_re HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47 \
  PRODUCTS="l0|$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni|0.85|131072 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85|131072" \
  setsid nohup bash "$REPO/chain_gpqa_1019.sh" F >> "$OUT/gpqa_re_F.log" 2>&1 < /dev/null &
env LANE=G CFGSUF=_re HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
  PRODUCTS="p2clip|$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85|131072 w8a8|$W8/glm5-w8a8_replica_p2clipfold_w8expert|0.95|69632" \
  setsid nohup bash "$REPO/chain_gpqa_1019.sh" G >> "$OUT/gpqa_re_G.log" 2>&1 < /dev/null &
for i in $(seq 1 96); do
  [ -f "$OUT/gpqa_l0_re_DONE" ] && [ -f "$OUT/gpqa_w4a8c8_re_DONE" ] && \
  [ -f "$OUT/gpqa_p2clip_re_DONE" ] && [ -f "$OUT/gpqa_w8a8_re_DONE" ] && break
  sleep 300
done
if [ -f "$OUT/gpqa_l0_re_DONE" ] && [ -f "$OUT/gpqa_w4a8c8_re_DONE" ] && \
   [ -f "$OUT/gpqa_p2clip_re_DONE" ] && [ -f "$OUT/gpqa_w8a8_re_DONE" ]; then
  touch "$OUT/GPQA_RE_ALL_DONE"
  log "stage3 complete - GPQA_RE_ALL_DONE"
else
  log "stage3 incomplete (see gpqa_re_*.log)"
fi
log "NIGHT DONE"
