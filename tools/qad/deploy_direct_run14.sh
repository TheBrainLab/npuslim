#!/bin/bash
# deploy_direct_run14.sh — 直接在 super2(HEAD)+super4(WORKER) 上部署
set -e

REENC="/llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600"
SERVED="glm5-w4a8-reenc-run14-s600"
CONTAINER=e6_noawq_env
PORT=8000
RAY_PORT=6379
IFNAME=enp66s0f1

# 需要在 HEAD_HOST 上执行：停旧服务 + 起 ray head + vllm serve
HEAD_HOST=yult@10.1.0.106
WORKER_HOST=yult@10.1.0.47

ssh_q() { timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

echo "== step 1: stop stale =="
for h in "$WORKER_HOST" "$HEAD_HOST"; do
  ssh_q "$h" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"vllm[/]entrypoints\"; ray stop --force >/dev/null 2>&1; pkill -9 -x raylet; pkill -9 -x gcs_server; pkill -9 -f \"ray[::]\"; sleep 3' >/dev/null 2>&1" || true
done
echo " stopped."

echo "== step 2: ray head on super2 =="
ssh_q "$HEAD_HOST" "docker exec $CONTAINER bash -lc \"ray start --head --port=${RAY_PORT} --node-ip-address=10.1.0.106 --num-cpus=160 --num-gpus=8 --min-worker-port=20000 --max-worker-port=29999 >/tmp/ray_head.log 2>&1\""
sleep 8
ssh_q "$HEAD_HOST" "docker exec $CONTAINER bash -lc 'ray status | head -12'"

echo "== step 3: ray worker on super4 =="
ssh_q "$WORKER_HOST" "docker exec $CONTAINER bash -lc \"ray start --address='10.1.0.106:${RAY_PORT}' --node-ip-address=10.1.0.47 --num-cpus=160 --num-gpus=8 >/tmp/ray_worker.log 2>&1\""
sleep 8
ssh_q "$HEAD_HOST" "docker exec $CONTAINER bash -lc 'ray status | head -12'"

echo "== step 4: vllm serve =="
ENVS="-e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NPUSLIM_PLUGIN_ENABLE=1 -e GLOO_SOCKET_IFNAME=$IFNAME -e HCCL_SOCKET_IFNAME=$IFNAME -e TP_SOCKET_IFNAME=$IFNAME -e VLLM_HOST_IP=10.1.0.106 -e VLLM_RPC_TIMEOUT=360000 -e HCCL_EXEC_TIMEOUT=2000"
CMD="vllm serve $REENC --served-model-name $SERVED --tensor-parallel-size 16 --distributed-executor-backend ray --quantization ascend --enable-expert-parallel --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.85 --block-size 128 --max-num-batched-tokens 32768 --max-num-seqs 64 --enable-chunked-prefill --host 0.0.0.0 --port $PORT"

ssh_q "$HEAD_HOST" "docker exec -d $ENVS $CONTAINER bash -lc 'nohup $CMD > /tmp/vllm_reenc_run14_s600.log 2>&1'"

echo "vLLM launching... wait 5min then check:"
echo "  curl -s http://10.1.0.106:8000/health"
