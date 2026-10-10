#!/bin/bash
# deploy_run14_step600_v2.sh — 手动部署 run14 reencode 产物到 super2(HEAD)+super4(WORKER)
set -e

REENC="/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600"
SERVED="glm5-w4a8-reenc-run14-s600"
HEAD_IP=10.1.0.106
WORKER_IP=10.1.0.47
HEAD_HOST=yult@10.1.0.106
WORKER_HOST=yult@10.1.0.47
CONTAINER=e6_noawq_env
PORT=8000
RAY_PORT=6379
IFNAME=enp66s0f1
SHARED=/llm_infer_workspace

ssh_q() { timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }
ssh_l() { timeout 400 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

echo "== [1/4] stop stale =="
for h in "$WORKER_HOST" "$HEAD_HOST"; do
  ssh_q "$h" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"vllm[/]entrypoints\"; ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; sleep 2' >/dev/null 2>&1" || true
done
sleep 3
echo "  stale processes stopped"

echo "== [2/4] layer_types patch =="
for h in "$HEAD_HOST" "$WORKER_HOST"; do
  ssh_q "$h" "docker exec ${CONTAINER} python3 -c \"
import json
p='${REENC}/config.json'
c=json.load(open(p))
n=len(c['layer_types'])
c['layer_types']=['sparse']*n
json.dump(c,open(p,'w'),indent=2,ensure_ascii=False)
print('  layer_types patched: sparse x',n)\""
done

echo "== [3/4] ray cluster =="
COMMON_ENV_ARGS="-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NPUSLIM_PLUGIN_ENABLE=1 -e GLOO_SOCKET_IFNAME=${IFNAME} -e HCCL_SOCKET_IFNAME=${IFNAME} -e TP_SOCKET_IFNAME=${IFNAME} -e HCCL_INTRA_PCIE_ENABLE=1 -e HCCL_INTRA_ROCE_ENABLE=0 -e HCCL_BUFFSIZE=512 -e HCCL_OP_EXPANSION_MODE=AIV -e HCCL_IF_BASE_PORT=60000 -e TASK_QUEUE_ENABLE=1 -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1 -e VLLM_RPC_TIMEOUT=360000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 -e HCCL_EXEC_TIMEOUT=2000 -e HCCL_CONNECT_TIMEOUT=1200"

# Start ray head
ssh_q "$HEAD_HOST" "docker exec ${COMMON_ENV_ARGS} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} \
  bash -lc \"ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --dashboard-host=0.0.0.0 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084 >/tmp/ray_head.log 2>&1\""
sleep 6

# Start ray worker
ssh_q "$WORKER_HOST" "docker exec ${COMMON_ENV_ARGS} -e VLLM_HOST_IP=${WORKER_IP} ${CONTAINER} \
  bash -lc \"ray start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address=${WORKER_IP} --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 --runtime-env-agent-port=18080 --dashboard-agent-listen-port=18081 --dashboard-agent-grpc-port=18082 --ray-client-server-port=18083 --metrics-export-port=18084 >/tmp/ray_worker.log 2>&1\""
sleep 6

# Show cluster
ssh_q "$HEAD_HOST" "docker exec ${CONTAINER} bash -lc 'ray status | head -12'"

echo "== [4/4] vllm serve =="
serve_cmd="vllm serve ${REENC} \
  --served-model-name ${SERVED} \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --quantization ascend \
  --enable-expert-parallel \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 64 \
  --enable-chunked-prefill \
  --host 0.0.0.0 \
  --port ${PORT}"

ssh_q "$HEAD_HOST" "docker exec -d ${COMMON_ENV_ARGS} -e VLLM_HOST_IP=${HEAD_IP} ${CONTAINER} \
  bash -lc 'nohup ${serve_cmd} > /tmp/vllm_reenc_run14_s600.log 2>&1'"

echo "  vLLM launching... check health:"
echo "    curl -s http://${HEAD_IP}:${PORT}/health"
echo ""

echo "== Deploy started =="
