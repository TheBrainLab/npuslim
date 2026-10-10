#!/bin/bash
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
: "${SHARED:=/llm_infer_workspace}"
: "${SHARED_HOST:=/home/yult/llm_infer_workspace}"
# =========================================================
# deploy_eval_artifact.sh — 部署一个产物 + 快评(wikitext+humaneval)；需在 HEAD 节点运行
# required env: MODEL MODEL_HOST ART_DIR SERVED WORKER_IP WORKER_HOST HEAD_IP HEAD_HOST OUT MARK_DIR TAG
set -uo pipefail
H=${SHARED_HOST}
DEPD=$H/ylt/npuslim_repo/tools/deploy
LOG=$H/ylt/glm5.2_deploy/deploy_eval_${TAG}.log
echo "[$TAG $(date '+%F %T')] deploy started" >> $LOG

env MODEL=$MODEL MODEL_HOST=$MODEL_HOST ART_DIR=$ART_DIR SERVED=$SERVED \
    WORKER_IP=$WORKER_IP WORKER_HOST=$WORKER_HOST HEAD_IP=$HEAD_IP HEAD_HOST=$HEAD_HOST \
    bash $DEPD/run_deploy_glm5_w4a8_artifact.sh >> $LOG 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[$TAG $(date '+%F %T')] DEPLOY FAILED rc=$rc" >> $LOG; exit 1; fi
echo "[$TAG $(date '+%F %T')] deploy OK -> fast eval (wikitext+humaneval, round1 pre-skipped)" >> $LOG

rm -rf $MARK_DIR; mkdir -p $MARK_DIR
if [ "${ROUNDS:-fast}" = "mmlu" ]; then
  # 只跑 round1（mmlu+gsm8k）
  touch $MARK_DIR/round2 $MARK_DIR/round3
else
  # 只跑 round2+round3（wikitext ppl + humaneval 快评）
  touch $MARK_DIR/round1
fi
env PORT=8000 HEAD_IP=$HEAD_IP MODEL_DIR=$MODEL OUT=$OUT MARK_DIR=$MARK_DIR \
    bash $DEPD/run_glm5_l15_tier1_eval.sh >> $LOG 2>&1
rc2=$?
echo "[$TAG $(date '+%F %T')] EVAL DONE rc=$rc2" >> $LOG

# 评测后停服务，释放节点对给后续实验
env MODEL=$MODEL MODEL_HOST=$MODEL_HOST ART_DIR=$ART_DIR SERVED=$SERVED \
    WORKER_IP=$WORKER_IP WORKER_HOST=$WORKER_HOST HEAD_IP=$HEAD_IP HEAD_HOST=$HEAD_HOST \
    bash $DEPD/run_deploy_glm5_w4a8_artifact.sh --stop >> $LOG 2>&1
echo "[$TAG $(date '+%F %T')] server stopped (pair released)" >> $LOG
