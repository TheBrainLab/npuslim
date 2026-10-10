#!/bin/bash
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
: "${SHARED:=/llm_infer_workspace}"
: "${SHARED_HOST:=/home/yult/llm_infer_workspace}"
# =========================================================
# run_deploy_glm52_bf16_c3.sh - Deploy GLM-5.2 BF16 baseline on s1/s2/s3/s4 (10.1.0.105/106/57/47).
# TP=32 + expert-parallel via Ray (s1=head+APIServer, s2/s3/s4=workers, 8 NPUs each).
# Rebased from run_deploy_glm52_bf16_4node.sh (C3f super0-3) for C3 nodes (10.1.0.x, enp66s0f1):
#   - model -> BF16 snapshots/master, drop --quantization ascend
#   - container -> glm5_bf16_env on image vllm-ascend-npuslim:v0.23.0rc1-a3 (self-contained)
#   - runs ON s1 (inter-node passwordless SSH configured in shared /home/yult/.ssh)
#   - fresh Ray cluster (s1 head:6379), NOT reusing any stale cluster
#   - mnt=2048 HARD LIMIT (tried 4096 on 2026-09-03: NPU OOM - gather_logprobs needs
#     1.75GiB but BF16 weights pin memory so only 1.76GiB is free -> EngineCore crash.
#     Same lesson as W4A8_修改总结 §15.2. mnt CANNOT be raised on BF16.)
# PITFALLS AVOIDED (today's W4A8 C3 run):
#   - NO HCCL_IF_IP (caused "HCCL_SOCKET_IFNAME invalid" on worker)
#   - stale raylet/gcs_server killed before ray start (residual env from old cluster)
#   - pkill uses regex char-class [..] to avoid matching the invoking shell itself
#   - health always checked on s1 localhost (local box cannot reach 10.1.0.105:9081 directly)
#
# Run ON s1 (HOST path for the script! ${SHARED} only exists inside containers):
#   ssh C3_super1 'nohup bash ${SHARED_HOST}/ylt/glm5.2_deploy/run_deploy_glm52_bf16_c3.sh > /tmp/deploy_glm52_bf16_c3.log 2>&1 &'
# Stop (from s1):
#   ssh C3_super1 'bash ${SHARED_HOST}/ylt/glm5.2_deploy/run_deploy_glm52_bf16_c3.sh --stop'
set -uo pipefail

PORT="${PORT:-9081}"
MODEL="${SHARED}/models/GLM-5.2-bf16/snapshots/master"   # container-side path (vllm serve)
MODEL_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}/models/GLM-5.2-bf16/snapshots/master"  # host-side path (preflight on s1 host)
SERVED="glm5-bf16-4node"
HEAD_IP="10.1.0.106"
NODE_IPS=("10.1.0.106" "10.1.0.57" "10.1.0.47" "10.1.0.135")
CONTAINER="glm5_bf16_env"
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
SHARED_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}"
SHARED="${SHARED:-/llm_infer_workspace}"
IFNAME="enp66s0f1"                                     # 10.1.0.x iface on all 4 C3 nodes
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MEM_UTIL="${MEM_UTIL:-0.9}"
TP=32
LOG_INNER="/tmp/vllm_glm52_bf16.log"
RAY_PORT=6379

SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes)
ssh_q() { timeout 60 ssh "${SSH_OPTS[@]}" "$@" 2>/dev/null; }
ssh_l() { timeout 600 ssh "${SSH_OPTS[@]}" "$@" 2>/dev/null; }  # docker run / ray start

health_on_head() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo 000
}

stop_remote() {
  # regex char-class avoids pkill -f matching the invoking shell's own cmdline
  ssh_q "$1" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"vllm[/]entrypoints\"; ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; sleep 2' >/dev/null 2>&1" || true
}

if [ "${1:-}" = "--stop" ]; then
  echo "== stopping GLM-5.2 BF16 on all 4 nodes =="
  for ip in "${NODE_IPS[@]}"; do echo "  stop ${ip}"; stop_remote "$ip"; done
  echo "stopped."
  exit 0
fi

echo "== [0/5] preflight (model + image on all nodes) =="
[ -d "$MODEL_HOST" ] || { echo "ERROR: model not found (local s1 host): $MODEL_HOST" >&2; exit 1; }
for ip in "${NODE_IPS[@]}"; do
  ssh_q "$ip" "docker image inspect ${IMAGE} >/dev/null 2>&1" || { echo "ERROR: image ${IMAGE} missing on ${ip} - docker load -i ${SHARED_HOST}/ylt/vllm-ascend-npuslim.tar" >&2; exit 1; }
done
# PITFALL (today): leftover NPU-bound containers (e.g. glm5_w4a8_env) keep the davinci
# devices "used" (dcmi -8020), so aclrtGetSocName()==NULL inside THIS container ->
# libvllm_ascend_kernels.so ctor does std::string(NULL) -> worker Abort on import.
# Stop them here (W4A8 eval already finished) before proceeding.
for ip in "${NODE_IPS[@]}"; do
  if ssh_q "$ip" "docker ps --format '{{.Names}}' | grep -qx 'glm5_w4a8_env'"; then
    echo "WARNING: glm5_w4a8_env still running on ${ip} (holds NPUs) - stopping it"
    ssh_q "$ip" "docker stop glm5_w4a8_env >/dev/null 2>&1" || true
  fi
done
echo "preflight OK (model on s1 + image on all 4 nodes)"

code="$(health_on_head)"
if [ "$code" = "200" ]; then
  echo "Server already up at http://${HEAD_IP}:${PORT}/health - reusing."; exit 0
fi

echo "== [1/5] stop stale vllm/ray on all nodes =="
for ip in "${NODE_IPS[@]}"; do stop_remote "$ip"; done
sleep 3
for ip in "${NODE_IPS[@]}"; do
  ssh_q "$ip" "docker exec ${CONTAINER} bash -lc 'pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; pkill -9 -f \"vllm[/]entrypoints\"; sleep 1' >/dev/null 2>&1" || true
done
sleep 2

# ---------------------------------------------------------------- container ----
echo "== [2/5] bring up container ${CONTAINER} on all 4 nodes =="
create_container_remote() {
  ssh_l "$1" "bash -s" <<'INNER'
set -e
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
CONTAINER="glm5_bf16_env"
SHARED_HOST="${SHARED_HOST:-/home/yult/llm_infer_workspace}"
SHARED="${SHARED:-/llm_infer_workspace}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "  ERROR: image missing: $IMAGE"; exit 1; }
state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo absent)
case "$state" in
  running) echo "  container running - skip"; exit 0 ;;
  absent)  : ;;
  *) echo "  stale container (${state}) - recreate"; docker rm -f "$CONTAINER" >/dev/null 2>&1 ;;
esac
DEV=""
for i in 0 1 2 3 4 5 6 7; do [ -e "/dev/davinci${i}" ] && DEV="$DEV --device=/dev/davinci${i}"; done
for d in davinci_manager devmm_svm hisi_hdc; do [ -e "/dev/${d}" ] && DEV="$DEV --device=/dev/${d}"; done
MNT=""
for p in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi \
         /usr/local/sbin/npu-smi /usr/local/bin/npu-smi \
         /usr/local/Ascend/driver/tools/hccn_tool /usr/local/Ascend/driver/version.info \
         /etc/ascend_install.info /etc/hccn.conf; do
  [ -e "$p" ] && MNT="$MNT -v $p:$p:ro"
done
[ -e /var/log/npu ] && MNT="$MNT -v /var/log/npu:/var/log/npu"
MNT="$MNT -v ${SHARED_HOST}:${SHARED}"
docker run -d --name "$CONTAINER" --restart unless-stopped --net=host --shm-size=10g \
  $DEV $MNT \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NPUSLIM_PLUGIN_ENABLE=1 \
  -e HF_HOME=${SHARED}/hf_home \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w "${SHARED}/ylt" \
  "$IMAGE" /bin/bash -lc "sleep infinity" >/dev/null 2>&1
echo "  container started on $(hostname)"
INNER
}
for ip in "${NODE_IPS[@]}"; do
  echo "  create on ${ip}..."
  create_container_remote "$ip" || { echo "  FAILED on ${ip}" >&2; exit 1; }
done

# ---------------------------------------------------------------- ray --------
echo "== [3/5] ray cluster (head=${HEAD_IP}:${RAY_PORT}, workers s2/s3/s4) =="
COMMON_ENV=(-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e NPUSLIM_PLUGIN_ENABLE=1
    -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME}
    -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0
    -e HCCL_BUFFSIZE=512 -e HCCL_OP_EXPANSION_MODE=AIV
    -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200)

if docker exec "${COMMON_ENV[@]}" -e VLLM_HOST_IP=${HEAD_IP} "${CONTAINER}" \
     bash -lc "ray status >/dev/null 2>&1" 2>/dev/null; then
  echo "  ray head already up - reuse"
else
  # PITFALL (today): Ray auto-picks random ports for runtime_env_agent/dashboard_agent/
  # metrics_export; they can collide with worker_ports [20000,29999] -> "worker_ports is
  # trying to use a port number ... used by other components". Pin ALL of them OUTSIDE
  # that range explicitly (18080/18081/18082/18083/18084).
  docker exec "${COMMON_ENV[@]}" -e VLLM_HOST_IP=${HEAD_IP} "${CONTAINER}" \
    bash -lc "ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f 'ray[::]'; sleep 1; ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084 >/tmp/ray_head.log 2>&1" \
    || { echo "  ray head FAILED" >&2; exit 1; }
fi
sleep 6

for ip in "${NODE_IPS[@]:1}"; do
  if ssh_q "$ip" "docker exec ${CONTAINER} bash -lc 'ray status >/dev/null 2>&1'"; then
    echo "  ray worker ${ip} already up - reuse"
    continue
  fi
  ssh_q "$ip" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${ip} ${CONTAINER} \
      bash -lc \"ray start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address=${ip} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --metrics-export-port=18084 >/tmp/ray_worker.log 2>&1\"" \
    || { echo "  ray worker ${ip} FAILED" >&2; exit 1; }
  sleep 3
done
sleep 6
echo "  --- ray status ---"
docker exec "${CONTAINER}" bash -lc 'ray status | head -14'

# ---------------------------------------------------------------- serve ------
echo "== [4/5] vllm serve (TP=${TP}, EP, BF16, mnt=2048) on ${HEAD_IP}:${PORT} =="
serve_cmd="vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --tensor-parallel-size ${TP} \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --block-size 128 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --host 0.0.0.0 \
    --port ${PORT} 2>&1 | tee ${LOG_INNER}"
docker exec -d "${COMMON_ENV[@]}" -e VLLM_HOST_IP=${HEAD_IP} "${CONTAINER}" \
  bash -lc "nohup ${serve_cmd} > /tmp/vllm_glm52_bf16_nohup.log 2>&1" \
  || { echo "  vllm launch FAILED" >&2; exit 1; }

echo "  vLLM launching (1.4TB BF16, TP=32 ray). Monitor:"
echo "    docker exec ${CONTAINER} tail -f ${LOG_INNER}"
echo "    health: curl -s localhost:${PORT}/health"

echo "== [5/5] waiting for health (max 50 min) =="
UP=0
for i in $(seq 1 100); do
    sleep 30
    code="$(health_on_head)"
    echo "  [$(printf '%03d' $((i*30)))s] health: ${code}"
    if [ "$code" = "200" ]; then UP=1; echo "SERVER UP"; break; fi
done
if [ "$UP" = "1" ]; then
    echo "ALL READY at http://${HEAD_IP}:${PORT}"
else
    echo "TIMEOUT/FAILED - check ${LOG_INNER}"
    docker exec "${CONTAINER}" bash -lc "tail -n 60 ${LOG_INNER}"
    exit 1
fi
