#!/bin/bash

# ===== Phase D 路径参数化（env 带默认回退）=====
: "${SHARED:=/llm_infer_workspace}"
: "${SHARED_HOST:=/home/yult/llm_infer_workspace}"
: "${YLT:=$SHARED/ylt}"
: "${NPUSLIM_SRC:=$YLT/npuslim/src}"
: "${QAD_ROOT:=$YLT/glm52-qad}"
# ==============================================
# diag_launch.sh — QAD_DIAG=1 diagnostic run (backward-degradation localization)
#
# Runs the production trainer with diag_ep instrumentation on super2/3/4/5 (world=32).
# Usage (from THIS machine): QAD_DIAG=1 STEPS=16 bash diag_launch.sh <node_rank 0-3>
#   STEPS        default 16
#   MICRO_TOKENS default 512
#   LOG_PREFIX   default diag32  (logs ${LOG_PREFIX}_r{N}.log)
#   OFFLOAD_PARAM default 1 ; QAD_EP default 1 ; STRIP_DS_HOOKS default 1 ; QAD_PROFILE default 1
set -uo pipefail

MODE=${1:?usage: diag_launch.sh <node_rank 0-3>}
STEPS=${STEPS:-16}
MICRO_TOKENS=${MICRO_TOKENS:-512}
LOG_PREFIX=${LOG_PREFIX:-diag32}
SAVE_EVERY=${SAVE_EVERY:-0}
SAVE_DIR=${SAVE_DIR:-ckpt_diag}

PORT=29516
H=${SHARED_HOST}
SH=${SHARED}
DEP=$SH/ylt/qad_deps/pkg
ASSET=$SH/ylt/glm52-qad
ARTIFACT=${ARTIFACT:-$SH/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234}

COMMON_ENV="PYTHONPATH=$DEP OMP_NUM_THREADS=32 ZERO3_PERSIST_THRESHOLD=0 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OFFLOAD_PARAM=${OFFLOAD_PARAM:-1} \
GLM_MODEL_DIR=$SH/models/GLM-5.2-bf16/snapshots/master \
GLM_DEPLOY_SCALE_DIR=$ARTIFACT \
MEM_WATCHDOG_GIB=${MEM_WATCHDOG_GIB:-80} \
HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600} \
    QAD_PROFILE=${QAD_PROFILE:-1} \
    QAD_EP=${QAD_EP:-1} \
    STRIP_DS_HOOKS=${STRIP_DS_HOOKS:-1} \
    QAD_DIAG=${QAD_DIAG:-1} \
    QAD_TPROF=${QAD_TPROF:-0} \
    QAD_TPROF_WAIT=${QAD_TPROF_WAIT:-3} \
    QAD_TPROF_ACTIVE=${QAD_TPROF_ACTIVE:-2} \
    QAD_TPROF_OUT=${QAD_TPROF_OUT:-tprof_r0.json} \
    QAD_COMPOSE=${QAD_COMPOSE:-post} \
    QAD_PREFETCH=${QAD_PREFETCH:-0}"

TRAIN_ARGS="--steps $STEPS --micro-tokens $MICRO_TOKENS --rank 16 --save-every $SAVE_EVERY --save-dir $SAVE_DIR \
--init-mode stream --frozen-source artifact --artifact-dir $ARTIFACT \
--skip-fake-quant --opt-mode ${OPT_MODE:-adamw_bf16}"

PYTHON=$SH/ylt/venv_t512/bin/python

case $MODE in
    0) NODE=${NODE0:-C3_super2}; MASTER=10.1.0.106;;
    1) NODE=${NODE1:-C3_super3}; MASTER=10.1.0.106;;
    2) NODE=${NODE2:-C3_super4}; MASTER=10.1.0.106;;
    3) NODE=${NODE3:-C3_super5}; MASTER=10.1.0.106;;
    *) echo "Unknown node_rank: $MODE (expect 0-3)"; exit 1;;
esac
NODES_ALL="${NODE0:-C3_super2} ${NODE1:-C3_super3} ${NODE2:-C3_super4} ${NODE3:-C3_super5}"

LOGNAME=${LOG_PREFIX}_r${MODE}
host_log=$H/ylt/${LOGNAME}.log
ctr_log=$SH/ylt/${LOGNAME}.log
mem_log=$H/ylt/${LOGNAME}_hbm.log

echo "============================================================"
echo "[diag] node_rank=$MODE ($NODE) STEPS=$STEPS MICRO_TOKENS=$MICRO_TOKENS QAD_DIAG=${QAD_DIAG:-1} world=32"
echo "  OFFLOAD_PARAM=${OFFLOAD_PARAM:-1} QAD_EP=${QAD_EP:-1} STRIP_DS_HOOKS=${STRIP_DS_HOOKS:-1}"
echo "  host_log=$host_log"
echo "============================================================"

if [ "$MODE" = "0" ]; then
  if ! PREFLIGHT_NODES="${PREFLIGHT_NODES:-$NODES_ALL}" bash "$(dirname "$0")/preflight_check.sh" "${PREFLIGHT_MIN_FREE:-48}"; then
    echo "[diag] REFUSE: 发射前资源门未通过（HBM 逐卡空闲不足或有占用者；见 0909 坑㊴）"
    echo "[diag] 调整门限：PREFLIGHT_MIN_FREE=<GiB> bash diag_launch.sh 0"
    exit 1
  fi
fi
cnt=$(ssh $NODE "docker exec e6_noawq_env pgrep -fc 'torch.distributed.run' 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
cnt=${cnt:-0}
echo "[preflight] $NODE torchrun count: $cnt (expect 0)"
if [ "$cnt" != "0" ]; then
    echo "[REFUSE] $NODE has torchrun running ($cnt). Clean up first."
    exit 1
fi

npu_busy=$(ssh $NODE "npu-smi info 2>/dev/null | sed -n '/Process id/,\$p' | grep -c '| python'" 2>/dev/null | tr -d '[:space:]')
npu_busy=${npu_busy:-0}
echo "[preflight] $NODE NPU python procs (npu-smi table): $npu_busy (expect 0)"
if [ "$npu_busy" != "0" ]; then
    echo "[REFUSE] $NODE has $npu_busy NPU process(es) holding cards."
    exit 1
fi

free_out=$(ssh $NODE "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null)
echo "[preflight] $NODE host avail: ${free_out}GiB"
if [ "${free_out:-0}" -lt 200 ]; then
    echo "[ABORT] $NODE host avail < 200GiB ($free_out GiB)."
    exit 1
fi

ssh $NODE 'docker exec e6_noawq_env bash -c "python3 -c \"import os; os.open('"'"'/dev/davinci0'"'"', os.O_RDWR); print('"'"'davinci0 OK'"'"')\""' 2>&1 | head -1

echo "[preflight] md5 gate on $NODE:"
ssh $NODE "md5sum ${SHARED_HOST}/ylt/glm52-qad/diag_ep.py ${SHARED_HOST}/ylt/glm52-qad/ds_train_glm.py" 2>/dev/null

ssh $NODE "setsid nohup bash -c 'while true; do printf \"%s \" \$(date +%H:%M:%S); npu-smi info 2>/dev/null | grep -oE \"[0-9]+ */ *65536\" | tr \"\n\" \" \"; echo; sleep 30; done' > $mem_log 2>&1 < /dev/null &" 2>/dev/null
echo "[diag] HBM sampler -> $mem_log"

ssh $NODE "docker exec -d e6_noawq_env bash -c 'cd $ASSET && echo \"[diag node_rank=$MODE] start \$(date)\" > $ctr_log && export $COMMON_ENV && setsid nohup $PYTHON -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank=$MODE --master_addr=$MASTER --master_port=$PORT ds_train_glm.py $TRAIN_ARGS >> $ctr_log 2>&1 < /dev/null'"
echo "[diag] torchrun dispatched on $NODE"
sleep 5
echo ""
echo "[diag] launched. Poll with:"
echo "  ssh $NODE 'grep -aE \"\\[diag step|^  step \" $ctr_log | tail -20'"
