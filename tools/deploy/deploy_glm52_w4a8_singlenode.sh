#!/bin/bash
# =============================================================================
# deploy_glm52_w4a8_singlenode.sh
#   Single-node GLM-5.2 W4A8 deploy on C3_super3 (10.1.0.57), TP=8 + EP.
#   Adapted from tools/deploy/run_deploy_glm5_w4a8_artifact.sh (dual-node TP=16)
#   to run a whole model replica on one 8-card node (the same topology zzw's
#   GLM-5.2-w4a8c8 DP replica uses: TP=8 + EP within a node).
#
#   env overrides: MODEL / SERVED / PORT / TP / MAX_MODEL_LEN / MEM_UTIL /
#                  NODE_HOST / CONTAINER / EXEC_BACKEND (mp|ray)
#   stop:  bash deploy_glm52_w4a8_singlenode.sh --stop
# =============================================================================
set -uo pipefail

NODE_HOST="${NODE_HOST:-10.1.0.57}"
NODE_IP="${NODE_IP:-10.1.0.57}"
CONTAINER="${CONTAINER:-e6_noawq_env}"
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
SHARED_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}"
SHARED="${SHARED:-/llm_infer_workspace}"
IFNAME="${IFNAME:-enp66s0f1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-${SHARED}/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni}"
MODEL_HOST="${MODEL_HOST:-${SHARED_HOST}${MODEL#${SHARED}}}"
SERVED="${SERVED:-glm52-w4a8-eval}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MEM_UTIL="${MEM_UTIL:-0.90}"
EXEC_BACKEND="${EXEC_BACKEND:-mp}"
RAY_PORT=6379
LOG_INNER="/tmp/vllm_${SERVED}.log"

ssh_q() { timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

health() { ssh_q "$NODE_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:${PORT}/health 2>/dev/null || echo 000"; }

stop_remote() {
  ssh_q "$NODE_HOST" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"[v]llm serve\"; pkill -9 -f \"[V]LLM::\"; ray stop --force >/dev/null 2>&1; pkill -9 -f \"[r]ay::\"; sleep 2' >/dev/null 2>&1" || true
}

if [ "${1:-}" = "--stop" ]; then
  echo "== stopping ${SERVED} on ${NODE_HOST} =="
  stop_remote
  echo "stopped."
  exit 0
fi

echo "== [0/5] preflight =="
ssh_q "$NODE_HOST" "[ -f ${MODEL_HOST}/quant_model_description.json ]" || { echo "ERROR: artifact missing: ${MODEL_HOST}" >&2; exit 1; }
ssh_q "$NODE_HOST" "docker image inspect ${IMAGE} >/dev/null 2>&1" || { echo "ERROR: image ${IMAGE} missing" >&2; exit 1; }
echo "preflight OK (${MODEL})"

code="$(health)"
if [ "$code" = "200" ]; then
  served_now="$(ssh_q "$NODE_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4")"
  if [ "$served_now" = "$SERVED" ]; then echo "Server already up (served=${SERVED}) - reusing."; exit 0; fi
  echo "WARNING: port ${PORT} occupied by '${served_now:-UNKNOWN}' != '${SERVED}' - stop first"; exit 1
fi

echo "== [1/5] stop stale vllm/ray =="
stop_remote
sleep 3

echo "== [2/5] start container ${CONTAINER} =="
ssh_q "$NODE_HOST" "docker start ${CONTAINER} >/dev/null 2>&1 || true"
sleep 4
ssh_q "$NODE_HOST" "docker exec ${CONTAINER} bash -lc 'echo container OK; nproc; python3 -c \"import vllm, torch_npu; print(\\\"vllm\\\", vllm.__version__)\"'"

echo "== [2b/5] runtime patches (idempotent) =="
for p in patch_deepseek_v2_loader.py patch_npu_fx_compiler.py; do
  ssh_q "$NODE_HOST" "docker exec ${CONTAINER} python3 ${SHARED}/ylt/npuslim/tools/serve/patches/${p} 2>&1 | tail -1 | sed 's/^/    [${p}] /'"
done

echo "== [2c/5] editable npuslim overlay =="
ssh_q "$NODE_HOST" "docker exec ${CONTAINER} bash -lc \"git config --global --add safe.directory '*' 2>/dev/null || true; NPUSLIM_SKIP_OPS=1 pip install --no-build-isolation --no-deps --root-user-action=ignore -e ${SHARED}/ylt/npuslim 2>&1 | tail -1; python3 -c 'import npuslim; print(\\\"  editable npuslim:\\\", npuslim.__file__)'\""

COMMON_ENV=(-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e NPUSLIM_PLUGIN_ENABLE=1
    -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME}
    -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0
    -e HCCL_BUFFSIZE=512 -e HCCL_OP_EXPANSION_MODE=AIV
    -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200)

if [ "$EXEC_BACKEND" = "ray" ]; then
  echo "== [3/5] ray head on ${NODE_IP} =="
  ssh_q "$NODE_HOST" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${NODE_IP} ${CONTAINER} bash -lc \"ray start --head --port=${RAY_PORT} --node-ip-address=${NODE_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 >/tmp/ray_head.log 2>&1\"" || { echo "  ray head FAILED" >&2; exit 1; }
  sleep 5
  BACKEND_ARG="--distributed-executor-backend ray"
else
  echo "== [3/5] mp backend (single node, no ray) =="
  BACKEND_ARG="--distributed-executor-backend mp"
fi

echo "== [4/5] vllm serve (TP=${TP}, EP, ascend quant) =="
serve_cmd="vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --tensor-parallel-size ${TP} \
    ${BACKEND_ARG} \
    --quantization ascend \
    --enable-expert-parallel \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --block-size 128 \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192} \
    --max-num-seqs ${MAX_NUM_SEQS:-32} \
    --enable-chunked-prefill \
    --return-tokens-as-token-ids \
    --host 0.0.0.0 \
    --port ${PORT} 2>&1 | tee ${LOG_INNER}"
# NOTE: '-d' + nohup -> detached on the node.
ssh_q "$NODE_HOST" "docker exec -d ${COMMON_ENV[@]} -e VLLM_HOST_IP=${NODE_IP} ${CONTAINER} bash -lc 'nohup ${serve_cmd} > ${LOG_INNER}.nohup 2>&1'" \
  || { echo "  vllm launch FAILED" >&2; exit 1; }

echo "== [5/5] waiting for health (max ~80 min) =="
UP=0
for i in $(seq 1 160); do
  sleep 30
  code="$(health)"
  [ $((i % 4)) -eq 0 ] && echo "  [$(printf '%04d' $((i*30)))s] health: ${code}"
  if [ "$code" = "200" ]; then UP=1; echo "SERVER UP"; break; fi
  # bail early if the serve process died
  if [ $((i % 4)) -eq 0 ]; then
    alive="$(ssh_q "$NODE_HOST" "docker exec ${CONTAINER} bash -lc 'pgrep -f \"[v]llm serve\" | head -1'")"
    if [ -z "$alive" ]; then echo "  vllm serve process gone - aborting (see ${LOG_INNER})"; exit 1; fi
  fi
done
if [ "$UP" = "1" ]; then
  echo "ALL READY at http://${NODE_IP}:${PORT} (served=${SERVED})"
  ssh_q "$NODE_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models | head -c 400"; echo
else
  echo "TIMEOUT/FAILED - check ${LOG_INNER}"
  exit 1
fi
