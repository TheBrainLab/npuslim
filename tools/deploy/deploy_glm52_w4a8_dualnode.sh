#!/bin/bash
# =============================================================================
# deploy_glm52_w4a8_dualnode.sh
#   Dual-node GLM-5.2 W4A8 deploy: HEAD=10.1.0.57, WORKER=10.1.0.47, TP=16 + EP,
#   ray backend.  Enables long context (>=128K) by halving per-card weight memory
#   and leaving HBM headroom for prompt-logprob tensors.
#   Adapted from tools/deploy/run_deploy_glm5_w4a8_artifact.sh (proven dual-node).
#
#   env: MODEL SERVED PORT TP MAX_MODEL_LEN MEM_UTIL MAX_NUM_BATCHED_TOKENS
#        MAX_NUM_SEQS HEAD_IP WORKER_IP CONTAINER
#   stop: bash deploy_glm52_w4a8_dualnode.sh --stop
# =============================================================================
set -uo pipefail

HEAD_IP="${HEAD_IP:-10.1.0.57}"
WORKER_IP="${WORKER_IP:-10.1.0.47}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"
WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
CONTAINER="${CONTAINER:-e6_noawq_env}"
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
SHARED_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}"
SHARED="${SHARED:-/llm_infer_workspace}"
IFNAME="${IFNAME:-enp66s0f1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-${SHARED}/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni}"
SERVED="${SERVED:-glm52-w4a8-eval}"
TP="${TP:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MEM_UTIL="${MEM_UTIL:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ROUTED_ARG=""
[ "${ROUTED_EXPERTS:-0}" = "1" ] && ROUTED_ARG="--enable-return-routed-experts"
[ "${ENFORCE_EAGER:-0}" = "1" ] && ROUTED_ARG="$ROUTED_ARG --enforce-eager"
RAY_PORT=6379
LOG_INNER="/tmp/vllm_${SERVED}.log"

ssh_q() { timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_l() { timeout 500 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
health() { ssh_q "$HEAD_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:${PORT}/health 2>/dev/null" | tail -1; }
stop_remote() {
  ssh_q "$1" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"[v]llm serve\"; pkill -9 -f \"[V]LLM::\"; ray stop --force >/dev/null 2>&1; pkill -9 -f \"[r]ay::\"; sleep 2' >/dev/null 2>&1" || true
}

if [ "${1:-}" = "--stop" ]; then
  echo "== stopping ${SERVED} on ${HEAD_HOST} + ${WORKER_HOST} =="
  stop_remote "$WORKER_HOST"; stop_remote "$HEAD_HOST"; echo stopped.; exit 0
fi

echo "== [0/6] preflight =="
MODEL_HOST="${SHARED_HOST}${MODEL#${SHARED}}"
for h in "$HEAD_HOST" "$WORKER_HOST"; do
  ssh_q "$h" "[ -f ${MODEL_HOST}/quant_model_description.json ]" || { echo "ERROR: artifact missing on $h" >&2; exit 1; }
  ssh_q "$h" "docker image inspect ${IMAGE} >/dev/null 2>&1" || { echo "ERROR: image missing on $h" >&2; exit 1; }
done
echo "preflight OK"

code="$(health)"
if [ "$code" = "200" ]; then
  sn="$(ssh_q "$HEAD_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4")"
  [ "$sn" = "$SERVED" ] && { echo "already up"; exit 0; }
  echo "port ${PORT} occupied by '${sn}' - stop first"; exit 1
fi

echo "== [1/6] stop stale on both =="
stop_remote "$WORKER_HOST"; stop_remote "$HEAD_HOST"; sleep 3

echo "== [2/6] container ${CONTAINER} on both =="
create_container_remote() {
  ssh_l "$1" "bash -s" <<INNER
set -e
IMAGE="$IMAGE"; CONTAINER="$CONTAINER"
SHARED_HOST="$SHARED_HOST"; SHARED="$SHARED"
docker image inspect "\$IMAGE" >/dev/null 2>&1 || { echo "  ERROR image missing"; exit 1; }
state=\$(docker inspect -f '{{.State.Status}}' "\$CONTAINER" 2>/dev/null || echo absent)
case "\$state" in
  running) echo "  container running - skip"; exit 0 ;;
  absent)  : ;;
  *) echo "  stale (\$state) - recreate"; docker rm -f "\$CONTAINER" >/dev/null 2>&1 ;;
esac
DEV=""
for i in 0 1 2 3 4 5 6 7; do [ -e "/dev/davinci\${i}" ] && DEV="\$DEV --device=/dev/davinci\${i}"; done
for d in davinci_manager devmm_svm hisi_hdc; do [ -e "/dev/\${d}" ] && DEV="\$DEV --device=/dev/\${d}"; done
MNT=""
for p in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi \
         /usr/local/sbin/npu-smi /usr/local/bin/npu-smi \
         /usr/local/Ascend/driver/tools/hccn_tool /usr/local/Ascend/driver/version.info \
         /etc/ascend_install.info /etc/hccn.conf; do
  [ -e "\$p" ] && MNT="\$MNT -v \$p:\$p:ro"
done
[ -e /var/log/npu ] && MNT="\$MNT -v /var/log/npu:/var/log/npu"
MNT="\$MNT -v \${SHARED_HOST}:\${SHARED}"
docker run -d --name "\$CONTAINER" --privileged=true --restart no --net=host --shm-size=10g \
  \$DEV \$MNT -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e OMP_NUM_THREADS=48 \
  -e HF_HOME=\${SHARED}/hf_home -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e RAY_worker_register_timeout_seconds=180 \
  -w "\${SHARED}/ylt" "\$IMAGE" /bin/bash -lc "sleep infinity" >/dev/null 2>&1
echo "  container started"
INNER
}
create_container_remote "$HEAD_HOST"   || { echo "container failed on HEAD" >&2; exit 1; }
create_container_remote "$WORKER_HOST" || { echo "container failed on WORKER" >&2; exit 1; }
ssh_q "$HEAD_HOST"   "docker start ${CONTAINER} >/dev/null 2>&1 || true"
ssh_q "$WORKER_HOST" "docker start ${CONTAINER} >/dev/null 2>&1 || true"
sleep 4

echo "== [3/6] patches + editable npuslim (both) =="
for h in "$HEAD_HOST" "$WORKER_HOST"; do
  for p in patch_deepseek_v2_loader.py patch_npu_fx_compiler.py patch_lm_eval_mbpp_utils_1016.py; do
    ssh_q "$h" "docker exec ${CONTAINER} python3 ${SHARED}/ylt/npuslim/tools/serve/patches/${p} 2>&1 | tail -1 | sed 's/^/    [${h} ${p}] /'"
  done
  ssh_q "$h" "docker exec ${CONTAINER} bash -lc \"git config --global --add safe.directory '*' 2>/dev/null || true; NPUSLIM_SKIP_OPS=1 pip install --no-build-isolation --no-deps --root-user-action=ignore -e ${SHARED}/ylt/npuslim 2>&1 | tail -1\""
  # Warm up Python byte-code caches (first import compiles them).  A freshly
  # created container otherwise makes the vLLM ray workers miss ray's
  # worker-registration timeout (observed 2026-10-16 on a recreated .47:
  # raylet "have not registered within the timeout" -> engine hung at 0/16 NPU).
  ssh_q "$h" "docker exec ${CONTAINER} bash -lc \"timeout 900 python3 -c 'import torch, torch_npu, ray, vllm, vllm_ascend' 2>&1 | tail -1 | sed 's/^/    [${h} warmup] /'\""
done

COMMON_ENV=(-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e NPUSLIM_PLUGIN_ENABLE=1
    -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME}
    -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0
    -e HCCL_BUFFSIZE=512 -e HCCL_OP_EXPANSION_MODE=AIV
    -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    -e NPUSLIM_ROUTER_DUMP=${ROUTER_DUMP:-}
    -e NPUSLIM_ROUTER_TOPK=${ROUTER_TOPK:-8}
    -e NPUSLIM_ROUTER_CALLS=${ROUTER_CALLS:-1}
    -e NPUSLIM_ROUTER_RANK=${ROUTER_RANK:-0}
    -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200)

echo "== [4/6] ray cluster (head=${HEAD_IP} worker=${WORKER_IP}) =="
RAY_EXTRA="--runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084"
ssh_q "$HEAD_HOST" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} bash -lc 'ray status >/dev/null 2>&1 || ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 ${RAY_EXTRA} >/tmp/ray_head.log 2>&1'" || { echo "ray head FAILED" >&2; exit 1; }
sleep 6
ssh_q "$WORKER_HOST" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${WORKER_IP} ${CONTAINER} bash -lc 'ray status >/dev/null 2>&1 || ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 ${RAY_EXTRA} >/tmp/ray_worker.log 2>&1'" || { echo "ray worker FAILED" >&2; exit 1; }
sleep 6
ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} bash -lc 'ray status | head -14'"

echo "== [5/6] vllm serve (TP=${TP}, EP, ray) =="
serve_cmd="vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --tensor-parallel-size ${TP} \
    --distributed-executor-backend ray \
    --quantization ascend \
    --enable-expert-parallel \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --block-size 128 \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --enable-chunked-prefill \
    --return-tokens-as-token-ids \
    ${ROUTED_ARG} \
    --host 0.0.0.0 --port ${PORT} 2>&1 | tee ${LOG_INNER}"
ssh_q "$HEAD_HOST" "docker exec -d ${COMMON_ENV[@]} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} bash -lc 'nohup ${serve_cmd} > ${LOG_INNER}.nohup 2>&1'" || { echo "vllm launch FAILED" >&2; exit 1; }

echo "== [6/6] waiting for health (max ~90 min) =="
UP=0
for i in $(seq 1 180); do
  sleep 30
  code="$(health)"
  [ $((i % 4)) -eq 0 ] && echo "  [$(printf '%04d' $((i*30)))s] health: ${code}"
  if [ "$code" = "200" ]; then UP=1; echo "SERVER UP"; break; fi
  if [ $((i % 4)) -eq 0 ]; then
    alive="$(ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} bash -lc 'pgrep -f \"[v]llm serve\" | head -1'")"
    [ -z "$alive" ] && { echo "vllm serve gone - abort (see ${LOG_INNER})"; exit 1; }
  fi
done
if [ "$UP" = "1" ]; then
  echo "ALL READY at http://${HEAD_IP}:${PORT} (served=${SERVED})"
  ssh_q "$HEAD_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models | head -c 300"; echo
else
  echo "TIMEOUT/FAILED"; exit 1
fi
