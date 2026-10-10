#!/bin/bash
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
SHARED=${SHARED:-}
SHARED_HOST=${SHARED_HOST:-/home/yult}
# =========================================================
# manual_eval_redeploy.sh — 手动重部署一个臂的评测（在 HEAD 节点运行）
# usage: manual_eval_redeploy.sh <HEAD_IP> <HEAD_HOST> <WORKER_IP> <WORKER_HOST> <ART> <TAG> [ROUNDS=fast|mmlu]
set -uo pipefail
HIP=$1; HHOST=$2; WIP=$3; WHOST=$4; ART=$5; TAG=$6; RND=${7:-fast}
H=/home/yult
SH=
DEPD=$H/ylt/npuslim/workspace/glm5.2_deploy
if [ "$RND" = "mmlu" ]; then OUTD=${ART}_round1; else OUTD=${ART}_fast; fi
echo "[redeploy $(date '+%F %T')] $TAG START art=$ART head=$HIP worker=$WIP rounds=$RND"
env ROUNDS=$RND MODEL=$SH/ylt/npuslim/outputs_npu/glm5/gptq/$ART \
    MODEL_HOST=$H/ylt/npuslim/outputs_npu/glm5/gptq/$ART \
    ART_DIR=$ART SERVED=glm5-w4a8-${TAG} \
    WORKER_IP=$WIP WORKER_HOST=$WHOST HEAD_IP=$HIP HEAD_HOST=$HHOST \
    OUT=$SH/ylt/glm5.2_deploy/outputs/$OUTD \
    MARK_DIR=/tmp/${TAG}_markers TAG=$TAG \
    bash $DEPD/deploy_eval_artifact.sh
rc=$?
echo "[redeploy $(date '+%F %T')] $TAG DONE rc=$rc"
exit $rc
