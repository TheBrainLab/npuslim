#!/bin/bash
# run_deploy_glm5_w4a8_artifact.sh - 通用 GLM-5.2 W4A8 产物双机部署（E6/E1-GLM/E2-GLM/E2b/E8 复用）
# on C3_super1 (head) + C3_super3 (worker), dual-node TP=16 + EP, ray backend.
# 参数（env 覆盖）: MODEL(容器内产物路径) / SERVED / CONTAINER
# 内置: devopen 预检（坑⑨）+ layer_types 自动 patch（坑⑪，原始 npuslim GLM 产物一律带雷）
# 注意（坑⑫）: 宿主侧 dtfs = /home/yult/llm_infer_workspace，容器内 = /llm_infer_workspace
# Run ON C3_super1 (它可 ssh 本机 localhost 与 super3 yult@10.1.0.57, 已验证):
#   ssh C3_super1 "setsid nohup env MODEL=... SERVED=... bash .../run_deploy_glm5_w4a8_artifact.sh > .../deploy_<tag>.log 2>&1 < /dev/null &"
# Stop:  run with --stop
set -uo pipefail

PORT=8000
MODEL="${MODEL:-/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_noawq}"
SERVED="${SERVED:-glm5-w4a8-noawq-e6}"
HEAD_IP="${HEAD_IP:-10.1.0.105}"       # C3_super1: ray head + APIServer
WORKER_IP="${WORKER_IP:-10.1.0.57}"    # default C3_super3; 可覆盖为 super2=10.1.0.106（super3 被他人占用时）
HEAD_HOST="${HEAD_HOST:-yult@10.1.0.105}"
WORKER_HOST="${WORKER_HOST:-yult@10.1.0.57}"
CONTAINER="${CONTAINER:-e6_noawq_env}"    # same container name on both nodes
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
SHARED_HOST="/home/yult/llm_infer_workspace"
SHARED="/llm_infer_workspace"
IFNAME="enp66s0f1"        # 10.1.0.x iface on super1/super3 (verified 09-10)
MAX_MODEL_LEN=8192
MEM_UTIL=0.85
NPUSLIM_DEV="${NPUSLIM_DEV:-1}"
LOG_INNER="/tmp/vllm_glm5_w4a8_noawq_e6.log"
RAY_PORT=6379

ssh_q() { timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_l() { timeout 400 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

health_on_head() {
  ssh_q "$HEAD_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:${PORT}/health 2>/dev/null || echo 000"
}

stop_remote() {
  ssh_q "$1" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"vllm[/]entrypoints\"; ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; sleep 2' >/dev/null 2>&1" || true
}

if [ "${1:-}" = "--stop" ]; then
  echo "== stopping E6 noAWQ deployment =="
  stop_remote "$WORKER_HOST"
  stop_remote "$HEAD_HOST"
  echo "stopped."
  exit 0
fi

echo "== [0/6] preflight =="
ssh_q "$HEAD_HOST" "[ -f ${SHARED_HOST}/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_noawq/quant_model_description.json ]" || { echo "ERROR: E6 model not found on ${HEAD_HOST}" >&2; exit 1; }
for h in "$HEAD_HOST" "$WORKER_HOST"; do
  ssh_q "$h" "docker image inspect ${IMAGE} >/dev/null 2>&1" || { echo "ERROR: image ${IMAGE} missing on ${h}" >&2; exit 1; }
  ssh_q "$h" "npu-smi info >/dev/null 2>&1" || { echo "ERROR: npu-smi failed on ${h}" >&2; exit 1; }
done
echo "preflight OK (artifact ${ART_DIR} + image + npu on both nodes)"

code="$(health_on_head)"
if [ "$code" = "200" ]; then
  served_now="$(ssh_q "$HEAD_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models 2>/dev/null | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4")"
  if [ "$served_now" = "$SERVED" ]; then echo "Server already up (served=${SERVED}) - reusing."; exit 0; fi
  echo "WARNING: port ${PORT} occupied by different served model (served=${served_now:-UNKNOWN} != target=${SERVED}) - stop and redeploy (坑⑱: health 不能单独判定复用)"
fi

echo "== [1/7] stop stale processes on both nodes =="
stop_remote "$WORKER_HOST"
stop_remote "$HEAD_HOST"
sleep 3

# ---------------------------------------------------------------- container ----
echo "== [2/7] bring up container ${CONTAINER} on both nodes =="
create_container_remote() {
  ssh_l "$1" "bash -s" <<'INNER'
set -e
IMAGE="vllm-ascend-npuslim:v0.23.0rc1-a3"
CONTAINER="e6_noawq_env"
SHARED_HOST="/home/yult/llm_infer_workspace"
SHARED="/llm_infer_workspace"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "  ERROR: image missing"; exit 1; }
state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo absent)
case "$state" in
  running) echo "  container running - skip create"; exit 0 ;;
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
docker run -d --name "$CONTAINER" --restart no --net=host --shm-size=10g \
  $DEV $MNT \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=48 \
  -e HF_HOME=${SHARED}/hf_home \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w "${SHARED}/ylt" \
  "$IMAGE" /bin/bash -lc "sleep infinity" >/dev/null 2>&1
echo "  container started"
# devopen 预检（坑⑨：EBUSY = UDA 驱动按 ns 登记使用中，定位持有容器后才能继续）
echo "  devopen precheck:"
devopen_rc=0
for i in 0 1 2 3 4 5 6 7; do
  if docker exec "$CONTAINER" python3 -c "import os; fd=os.open('/dev/davinci${i}', os.O_RDWR); os.close(fd)" 2>/dev/null; then
    echo "  davinci${i}: OK"
  else
    echo "  davinci${i}: EBUSY/FAIL (坑⑨: 定位 UDA 持有容器: root 读 /proc/uda/namespace_node → docker ps 匹配 → docker stop 持有者)" >&2
    devopen_rc=1
  fi
done
[ "$devopen_rc" = "0" ] || { echo "  ERROR: devopen precheck failed - aborting before ray/serve"; exit 1; }
echo "  applying runtime patches (idempotent)..."
for p in patch_deepseek_v2_loader.py patch_npu_fx_compiler.py; do
  docker exec "$CONTAINER" python3 "${SHARED}/ylt/npuslim/tools/serve/patches/${p}" 2>&1 | tail -1 | sed "s/^/    [${p}] /"
done
REPO_OK=0
for try in 1 2 3; do
  if [ -d "${SHARED}/ylt/npuslim" ]; then REPO_OK=1; break; fi
  echo "  shared repo not visible yet (try ${try}) - dtfs lazy mount?"
  sleep 10
done
if [ "$REPO_OK" = "1" ]; then
  echo "  NPUSLIM_DEV=1: editable-overlay npuslim from shared disk..."
  docker exec "$CONTAINER" bash -lc "git config --global --add safe.directory '*' 2>/dev/null || true; NPUSLIM_SKIP_OPS=1 pip install --no-build-isolation --no-deps --root-user-action=ignore -e ${SHARED}/ylt/npuslim 2>&1 | tail -1; python3 -c 'import npuslim; print(\"  editable npuslim from:\", npuslim.__file__)'"
else
  echo "  using self-contained npuslim from image"
fi
INNER
}
create_container_remote "$HEAD_HOST"   || { echo "ERROR: container setup failed on HEAD" >&2; exit 1; }
create_container_remote "$WORKER_HOST" || { echo "ERROR: container setup failed on WORKER" >&2; exit 1; }

# ---------------------------------------------------------------- config ----
echo "== [3/7] config precheck: layer_types auto-patch (坑⑪) =="
LT="$(ssh_q "$HEAD_HOST" "python3 -c \"import json;print(json.load(open('${MODEL_HOST}/config.json')).get('layer_types',[''])[0])\" 2>/dev/null" || echo UNKNOWN)"
if [ "$LT" = "deepseek_sparse_attention" ]; then
  echo "  layer_types=deepseek_sparse_attention (raw HF 值) - patching to 'sparse' via container(root)..."
  ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} python3 -c \"
import json
p='${MODEL}/config.json'
c=json.load(open(p))
n=len(c['layer_types'])
c['layer_types']=['sparse']*n
json.dump(c,open(p,'w'),indent=2,ensure_ascii=False)
print('  layer_types patched: sparse x',n)\"" || { echo "  ERROR: layer_types patch failed" >&2; exit 1; }
elif [ "$LT" = "sparse" ] || [ "$LT" = "" ]; then
  echo "  layer_types OK (${LT:-absent})"
else
  echo "  layer_types UNEXPECTED: ${LT} - 人工核对后再部署" >&2; exit 1
fi

# ---------------------------------------------------------------- ray --------
echo "== [4/7] ray cluster (head=${HEAD_IP} worker=${WORKER_IP}) =="
COMMON_ENV=(-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e NPUSLIM_PLUGIN_ENABLE=1
    -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME}
    -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0
    -e HCCL_BUFFSIZE=512 -e HCCL_OP_EXPANSION_MODE=AIV
    -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200)

if ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} bash -lc 'ray status >/dev/null 2>&1'"; then
  echo "  ray head already up - reuse"
else
  ssh_q "$HEAD_HOST" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} \
      bash -lc \"ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084 >/tmp/ray_head.log 2>&1\"" \
    || { echo "  ray head FAILED" >&2; exit 1; }
fi
sleep 6
if ssh_q "$WORKER_HOST" "docker exec ${CONTAINER} bash -lc 'ray status >/dev/null 2>&1'"; then
  echo "  ray worker already up - reuse"
else
  ssh_q "$WORKER_HOST" "docker exec ${COMMON_ENV[@]} -e VLLM_HOST_IP=${WORKER_IP} ${CONTAINER} \
      bash -lc \"ray start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address=${WORKER_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084 >/tmp/ray_worker.log 2>&1\"" \
    || { echo "  ray worker FAILED" >&2; exit 1; }
fi
sleep 6
ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} bash -lc 'ray status | head -12'"

# ---------------------------------------------------------------- serve ------
echo "== [4/6] vllm serve (TP=16, EP, ascend quant) on ${HEAD_IP} =="
serve_cmd="vllm serve ${MODEL} \
    --served-model-name ${SERVED} \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --quantization ascend \
    --enable-expert-parallel \
    --dtype auto \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${MEM_UTIL} \
    --block-size 128 \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-32768} \
    --max-num-seqs ${MAX_NUM_SEQS:-64} \
    --enable-chunked-prefill \
    --host 0.0.0.0 \
    --port ${PORT} 2>&1 | tee ${LOG_INNER}"
ssh_q "$HEAD_HOST" "docker exec -d ${COMMON_ENV[@]} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} \
    bash -lc 'nohup ${serve_cmd} > /tmp/vllm_glm5_w4a8_noawq_e6_nohup.log 2>&1'" \
  || { echo "  vllm launch FAILED" >&2; exit 1; }

echo "  vLLM launching (${ART_DIR}, TP=16 ray). Monitor:"
echo "    ssh C3_super1 \"docker exec ${CONTAINER} tail -f ${LOG_INNER}\""
echo "    health: curl -s http://${HEAD_IP}:${PORT}/health"

echo "== [5/6] waiting for health (max 50 min) =="
UP=0
for i in $(seq 1 100); do
    sleep 30
    code="$(health_on_head)"
    echo "  [$(printf '%03d' $((i*30)))s] health: ${code}"
    if [ "$code" = "200" ]; then UP=1; echo "SERVER UP"; break; fi
done
if [ "$UP" = "1" ]; then
    echo "ALL READY at http://${HEAD_IP}:${PORT} (served=${SERVED})"
else
    echo "TIMEOUT/FAILED - check ${LOG_INNER}"
    echo "  ssh C3_super1 \"docker exec ${CONTAINER} tail -n 60 ${LOG_INNER}\""
    exit 1
fi

echo "== [6/6] smoke: /v1/models =="
ssh_q "$HEAD_HOST" "curl -s --max-time 10 http://localhost:${PORT}/v1/models | head -c 400"
echo ""
echo "ARTIFACT DEPLOY DONE (${ART_DIR})"
