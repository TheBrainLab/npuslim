#!/bin/bash
# deploy_run14_step600.sh — 在 super2+super4 上部署 run14 reencode 产物
# 覆盖默认 HEAD=super1 为 super2，WORKER=super4
set -uo pipefail

REENC_DIR="/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600"
SERVED="glm5-w4a8-reenc-run14-s600"
DEPLOY_SCRIPT="/llm_infer_workspace/ylt/npuslim_repo/tools/deploy/run_deploy_glm5_w4a8_artifact.sh"

echo "== deploy run14 step600 =="
echo "HEAD=10.1.0.106 (super2)  WORKER=10.1.0.47 (super4)"
echo "MODEL=${REENC_DIR}"
echo ""

# 先停可能的旧服务
bash "$DEPLOY_SCRIPT" --stop 2>/dev/null || true
sleep 3

# 部署：super2 为 HEAD，super4 为 WORKER
env \
  HEAD_IP=10.1.0.106 \
  WORKER_IP=10.1.0.47 \
  HEAD_HOST=yult@10.1.0.106 \
  WORKER_HOST=yult@10.1.0.47 \
  MODEL="${REENC_DIR}" \
  SERVED="${SERVED}" \
  CONTAINER=e6_noawq_env \
  bash "$DEPLOY_SCRIPT" 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  echo ""
  echo "=== DEPLOY DONE ==="
  echo "API: http://10.1.0.106:8000/v1/completions"
  echo "Model: ${SERVED}"
  echo ""
  echo "Now run eval:"
  echo "  ssh 10.1.0.106 \"docker exec e6_noawq_env bash -c '"
  echo "    env PORT=8000 HEAD_IP=10.1.0.106 CONTAINER=e6_noawq_env OUT=/llm_infer_workspace/ylt/glm52-qad/eval_run14_step600"
  echo "      MODEL_DIR=${REENC_DIR} bash run_glm5_l15_tier1_eval.sh'\""
else
  echo "DEPLOY FAILED (rc=$rc)"
fi
