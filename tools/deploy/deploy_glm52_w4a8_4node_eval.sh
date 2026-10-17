#!/bin/bash
# =============================================================================
# deploy_glm52_bf16_4node_eval.sh
#   GLM-5.2 BF16 reference deployment for the L0 fidelity instruments + PPL-length.
#   4 nodes x 8 NPU, TP=32 + EP, ray backend.  Nodes (fixed): .106(head) .57 .47 .135
#   Adapted from run_deploy_glm52_bf16_s2345.sh with:
#     + --return-tokens-as-token-ids  (id-keyed logprobs for eval_dist_fidelity)
#     + --enable-chunked-prefill      (long-context PPL windows)
#     + env-parameterized MAX_MODEL_LEN / MEM_UTIL / MAX_NUM_BATCHED_TOKENS
#   NOTE: max-num-batched-tokens=2048 is a HARD limit on BF16 (weights pin HBM;
#         4096 -> gather_logprobs OOM, see script header of the original).
#   Run ON 10.1.0.106 (this host). Stop: --stop
# =============================================================================
set -uo pipefail
PORT="${PORT:-9081}"
SHARED_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}"
SHARED="${SHARED:-/llm_infer_workspace}"
MODEL="${MODEL:-${SHARED}/models/GLM-5.2-bf16/snapshots/master}"
MODEL_HOST="${SHARED_HOST}/models/GLM-5.2-bf16/snapshots/master"
SERVED="${SERVED:-glm52-bf16-eval}"
HEAD_IP="10.1.0.106"
NODE_IPS=("10.1.0.106" "10.1.0.57" "10.1.0.47" "10.1.0.135")
CONTAINER="${CONTAINER:-e6_noawq_env}"
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
IFNAME="enp66s0f1"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}"
MEM_UTIL="${MEM_UTIL:-0.95}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
USE_EP="${USE_EP:-1}"
EXTRA_ARGS=""
[ "$ENFORCE_EAGER" = "1" ] && EXTRA_ARGS="--enforce-eager"
EP_ARG=""
[ "$USE_EP" = "1" ] && EP_ARG="--enable-expert-parallel"
TP=32
LOG_INNER="/tmp/vllm_glm52_bf16.log"
RAY_PORT=6379

SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
ssh_q() { timeout 60 ssh "${SSH_OPTS[@]}" "$@" 2>/dev/null; }
ssh_l() { timeout 600 ssh "${SSH_OPTS[@]}" "$@" 2>/dev/null; }
health() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null; }
stop_remote() {
  ssh_q "$1" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"vllm[/]entrypoints\"; pkill -9 -f \"[V]LLM::\"; ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"[r]ay::\"; pkill -9 -x redis-server; sleep 2; rm -rf /tmp/ray' >/dev/null 2>&1" || true
}

if [ "${1:-}" = "--stop" ]; then
  echo "== stopping GLM-5.2 BF16 on all 4 nodes =="
  for ip in "${NODE_IPS[@]}"; do echo "  stop ${ip}"; stop_remote "$ip"; done
  echo stopped.; exit 0
fi

echo "== [0/5] preflight =="
[ -d "$MODEL_HOST" ] || { echo "ERROR: model missing: $MODEL_HOST" >&2; exit 1; }
for ip in "${NODE_IPS[@]}"; do
  ssh_q "$ip" "docker image inspect ${IMAGE} >/dev/null 2>&1" || { echo "ERROR: image missing on ${ip}" >&2; exit 1; }
done
echo "preflight OK"

code="$(health)"
if [ "$code" = "200" ]; then
  sn="$(curl -s --max-time 10 http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4)"
  [ "$sn" = "$SERVED" ] && { echo "already up (${SERVED})"; exit 0; }
  echo "port ${PORT} occupied by '${sn}' - stop first"; exit 1
fi

echo "== [1/5] stop stale =="
for ip in "${NODE_IPS[@]}"; do stop_remote "$ip"; done
sleep 3

echo "== [2/5] container ${CONTAINER} on 4 nodes =="
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
  \$DEV \$MNT -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NPUSLIM_PLUGIN_ENABLE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=\${SHARED}/hf_home -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e RAY_worker_register_timeout_seconds=180 \
  -w "\${SHARED}/ylt" "\$IMAGE" /bin/bash -lc "sleep infinity" >/dev/null 2>&1
echo "  container started on \$(hostname)"
INNER
}
for ip in "${NODE_IPS[@]}"; do create_container_remote "$ip" || { echo "container FAILED on $ip" >&2; exit 1; }; done

echo "== [2.5/5] npuslim + patches + warmup (all nodes) =="
for ip in "${NODE_IPS[@]}"; do
  for p in patch_deepseek_v2_loader.py patch_npu_fx_compiler.py patch_lm_eval_mbpp_utils_1016.py; do
    ssh_q "yult@${ip}" "docker exec ${CONTAINER} python3 ${SHARED}/ylt/npuslim/tools/serve/patches/${p} 2>&1 | tail -1 | sed 's/^/    [${ip} ${p}] /'"
  done
  ssh_q "yult@${ip}" "docker exec ${CONTAINER} bash -lc \"git config --global --add safe.directory '*' 2>/dev/null || true; NPUSLIM_SKIP_OPS=1 pip install --no-build-isolation --no-deps --root-user-action=ignore -e ${SHARED}/ylt/npuslim 2>&1 | tail -1\""
  ssh_q "yult@${ip}" "docker exec ${CONTAINER} bash -lc \"timeout 900 python3 -c 'import torch, torch_npu, ray, vllm, vllm_ascend' 2>&1 | tail -1 | sed 's/^/    [${ip} warmup] /'\""
done

COMMON_ENV=(-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e NPUSLIM_PLUGIN_ENABLE=1
    -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME}
    -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0
    -e HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512} -e HCCL_OP_EXPANSION_MODE=AIV
    -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=${FLASHCOMM:-1}
    -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200
    -e NPUSLIM_ROUTER_DUMP=${ROUTER_DUMP:-}
    -e NPUSLIM_ROUTER_TOPK=${ROUTER_TOPK:-8}
    -e NPUSLIM_ROUTER_CALLS=${ROUTER_CALLS:-1}
    -e NPUSLIM_ROUTER_RANK=${ROUTER_RANK:-0})

echo "== [3/5] ray cluster =="
RAY_EXTRA="--runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084"
docker exec "${COMMON_ENV[@]}" -e VLLM_HOST_IP=${HEAD_IP} "${CONTAINER}" \
  bash -lc "ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f 'ray[::]'; sleep 1; ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 ${RAY_EXTRA} >/tmp/ray_head.log 2>&1" \
  || { echo "ray head FAILED" >&2; exit 1; }
sleep 6
for ip in "${NODE_IPS[@]:1}"; do
  ssh_q "$ip" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${ip} ${CONTAINER} bash -lc 'ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; sleep 1; ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${ip} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 ${RAY_EXTRA} >/tmp/ray_worker.log 2>&1'" \
    || { echo "ray worker ${ip} FAILED" >&2; exit 1; }
  sleep 3
done
sleep 6
docker exec "${CONTAINER}" bash -lc 'ray status | head -12'

echo "== [4/5] vllm serve BF16 (TP=32, EP, mnt=${MAX_NUM_BATCHED_TOKENS}) on ${HEAD_IP}:${PORT} =="
serve_cmd="vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --tensor-parallel-size ${TP} \
    --distributed-executor-backend ray \
    ${EP_ARG} \
    --quantization ascend \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --block-size 128 \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --enable-chunked-prefill \
    --return-tokens-as-token-ids \
    ${EXTRA_ARGS} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port ${PORT} 2>&1 | tee ${LOG_INNER}"
docker exec -d "${COMMON_ENV[@]}" -e VLLM_HOST_IP=${HEAD_IP} "${CONTAINER}" \
  bash -lc "nohup ${serve_cmd} > /tmp/vllm_glm52_bf16_nohup.log 2>&1" \
  || { echo "vllm launch FAILED" >&2; exit 1; }

echo "== [5/5] waiting for health (max ~80 min) =="
UP=0
for i in $(seq 1 160); do
  sleep 30
  code="$(health)"
  [ $((i % 4)) -eq 0 ] && echo "  [$(printf '%04d' $((i*30)))s] health: ${code}"
  if [ "$code" = "200" ]; then UP=1; echo "SERVER UP"; break; fi
  if [ $((i % 4)) -eq 0 ]; then
    alive="$(docker exec ${CONTAINER} bash -lc 'pgrep -f "[v]llm serve" | head -1')"
    [ -z "$alive" ] && { echo "vllm gone - abort (see ${LOG_INNER})"; exit 1; }
  fi
done
if [ "$UP" = "1" ]; then
  echo "ALL READY at http://${HEAD_IP}:${PORT} (served=${SERVED})"
  curl -s --max-time 10 http://localhost:${PORT}/v1/models | head -c 300; echo
else
  echo "TIMEOUT/FAILED"; exit 1
fi
