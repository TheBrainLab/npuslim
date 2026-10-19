#!/bin/bash
# =============================================================================
# gpqa_launch_1019.sh — waits for the BF16 frontier arm, then runs the dedicated
# GPQA round (AISBench official recipe, full 198) on two lanes.
#   F: w4a8c8 -> l0      (official-alignment pair first)
#   G: w8a8   -> p2clip
# Idempotent: chain_gpqa_1019.sh skips arms with gpqa_<tag>_DONE markers, so a
# watchdog relaunch of this launcher resumes safely.
# =============================================================================
set -uo pipefail
cd /home/yult/llm_infer_workspace/ylt
OUT=glm52-qad/audit1009
REPO=npuslim_repo/tools/qad
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
W8=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq

for i in $(seq 1 240); do [ -f "$OUT/frontier_bf16_DONE" ] && break; sleep 300; done
sleep 60

env LANE=F HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47 \
  PRODUCTS="w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85|131072 l0|$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni|0.85|131072" \
  setsid nohup bash "$REPO/chain_gpqa_1019.sh" F >> "$OUT/gpqa_F_launch.log" 2>&1 < /dev/null &

env LANE=G HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
  PRODUCTS="w8a8|$W8/glm5-w8a8_replica_p2clipfold_w8expert|0.95|69632 p2clip|$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85|131072" \
  setsid nohup bash "$REPO/chain_gpqa_1019.sh" G >> "$OUT/gpqa_G_launch.log" 2>&1 < /dev/null &

wait
if [ -f "$OUT/GPQA_F_DONE" ] && [ -f "$OUT/GPQA_G_DONE" ]; then
  touch "$OUT/GPQA_ROUND_DONE"
fi

# --- appended: BF16 GPQA on 4 nodes (TP=32) after the products' round ---
if [ -f "$OUT/GPQA_ROUND_DONE" ] && [ ! -f "$OUT/gpqa_bf16_DONE" ]; then
  setsid nohup bash "$REPO/chain_gpqa_bf16_1019.sh" >> "$OUT/gpqa_bf16.log" 2>&1 < /dev/null &
  tries=0
  for i in $(seq 1 144); do
    [ -f "$OUT/gpqa_bf16_DONE" ] && break
    if ! pgrep -f 'chain_gpqa_bf16_1019.sh$' >/dev/null 2>&1; then
      tries=$((tries+1))
      if [ "$tries" -gt 3 ]; then echo "[gpqa-launch $(date '+%F %T')] bf16 gpqa failed 3x - giving up" >> "$OUT/gpqa_launch.log"; break; fi
      echo "[gpqa-launch $(date '+%F %T')] bf16 gpqa chain gone -> relaunch ($tries)" >> "$OUT/gpqa_launch.log"
      setsid nohup bash "$REPO/chain_gpqa_bf16_1019.sh" >> "$OUT/gpqa_bf16.log" 2>&1 < /dev/null &
      sleep 1800
    fi
    sleep 300
  done
fi
[ -f "$OUT/gpqa_bf16_DONE" ] && touch "$OUT/GPQA_ALL_DONE"
