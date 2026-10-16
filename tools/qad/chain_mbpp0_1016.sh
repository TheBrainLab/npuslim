#!/bin/bash
# =============================================================================
# chain_mbpp0_1016.sh — 0-prefix mbpp baseline (同协议：与 P2a 唯一的差别是
# 提示词不含 32K 前缀)。4 产物 × 2 节点（F 对 .57+.47，容器已新鲜+已打补丁）。
#   输出：glm52-qad/audit1009/lctx0_<tag>/
# 幂等由共享链保证；env: LANE（默认 M0）
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-M0}}"
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
REPO=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq
export TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp0
export OUT_TAG_PREFIX=lctx0
export HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47
export PRODUCTS="l0|${W4A8}/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 p2clip|${W4A8}/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85 w8a8|${REPO}/glm5-w8a8_replica_p2clipfold_w8expert|0.92"
exec bash /home/yult/llm_infer_workspace/ylt/npuslim_repo/tools/qad/chain_p2a_only.sh "$LANE"
