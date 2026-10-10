#!/bin/bash
# chain_qad_requant_deploy_eval.sh — STEP 4：等 QAD requant 完成 → 校验产物 → 部署(双机 TP=16+EP) → 四维评测
#
# 用法（本机前台调用；launcher 依赖 ~/.ssh/config）：
#   bash chain_qad_requant_deploy_eval.sh
#
# 模板：workspace/glm5.2_deploy/chain_wait_path2_deploy_eval.sh（同结构：等待/校验/释放/部署/评测）
# 评测协议：与底座四维同协议（run_glm5_l15_tier1_eval.sh，即 3.4507/0.8814/0.9454/0.9512 的口径）
#   round1 mmlu+gsm8k(5-shot, conc64, maxlen8192) / round2 wikitext(ppl@1024, conc2) / round3 humaneval(0-shot conc8)
#   ★ AGENTS.md #12 要求新产物先看 wikitext+humaneval：若需先出快数，把 EVAL 脚本的 round 顺序调整即可
#     （任务顺序不影响分值；本链默认沿用既有顺序以保证与基线完全同参）
set -uo pipefail

H=/home/yult/llm_infer_workspace
CH=$H/ylt/glm52-qad
REQ_LOG=$H/ylt/glm52-qad/requant_qad.log
ART=glm5-w4a8_modelslim_replica_layer1_5_path2_qad_requant
ART_DIR_HOST=$H/ylt/npuslim/outputs_npu/glm5/gptq/$ART
MODEL_CT=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/$ART
DEPLOY=$H/ylt/npuslim_repo/tools/deploy/run_deploy_glm5_w4a8_artifact.sh
EVAL=$H/ylt/npuslim_repo/tools/deploy/run_glm5_l15_tier1_eval.sh
# 评测节点：HEAD=super2(106) + WORKER=super4(47)（避开 super3 的 requant 卡0）
HI=10.1.0.106; HH=yult@10.1.0.106
WI=10.1.0.47;  WH=yult@10.1.0.47
SERVED=glm5-w4a8-p2qadreq
LOG_PREFIX=$CH/qadreq_chain
log() { echo "[qadreq-chain $(date '+%F %T')] $*"; }

log "1/5 等 requant 完成（timeout 40h，日志 $REQ_LOG）..."
for i in $(seq 1 2400); do
  grep -q "Pipeline completed" "$REQ_LOG" 2>/dev/null && break
  if [ $((i % 30)) -eq 0 ] && ! ssh -o BatchMode=yes yult@10.1.0.57 \
      "docker exec e6_noawq_env pgrep -f tools/run.py >/dev/null 2>&1"; then
    sleep 60; grep -q "Pipeline completed" "$REQ_LOG" 2>/dev/null && break
    log "ERROR: requant 进程消失且无完成标记"; exit 1
  fi
  sleep 60
done
grep -q "Pipeline completed" "$REQ_LOG" 2>/dev/null || { log "ERROR: requant 未在 40h 内完成"; exit 1; }
log "requant 完成"

IDX=$ART_DIR_HOST/model.safetensors.index.json
# 一致性校验：实际分片数 == index 引用分片数 + index 完整 + 体量合理
# （0924 修正：硬编码 NS>=100 误拒 requant 产物——Backfill 793 个 fp32 大张量改变切分，98 shards 即完整）
chk_prod() {
  NS=$(ls $ART_DIR_HOST/*.safetensors 2>/dev/null | wc -l)
  NS_IDX=$(grep -oE "model-[0-9]+\.safetensors" "$IDX" 2>/dev/null | sort -u | wc -l)
  TS=$(grep -oE '"total_size": [0-9]+' "$IDX" 2>/dev/null | grep -oE "[0-9]+")
  [ -f "$IDX" ] && [ "$NS" -ge 90 ] && [ "$NS" = "$NS_IDX" ] && [ "${TS:-0}" -ge 400000000000 ]
}
chk_prod
for k in $(seq 1 20); do chk_prod && break; sleep 15; chk_prod; done
if ! chk_prod; then
  NS=$(ls $ART_DIR_HOST/*.safetensors 2>/dev/null | wc -l)
  log "ERROR: 产物不完整（$NS shards, index=$NS_IDX, total_size=${TS:-0}）"; exit 1
fi
NS=$(ls $ART_DIR_HOST/*.safetensors 2>/dev/null | wc -l)
log "2/5 产物校验 OK: $NS shards（index 一致）total_size=$TS"

log "3/5 等 super3 卡0 释放（timeout 20min）..."
FREED=0
for j in $(seq 1 80); do
  if ssh -o BatchMode=yes yult@10.1.0.57 "docker exec e6_noawq_env pgrep -f tools/run.py >/dev/null 2>&1"; then
    sleep 15; else FREED=1; break; fi
done
[ "$FREED" = 1 ] || { log "ERROR: super3 卡0 未在 20min 内释放"; exit 1; }

log "4/5 部署到 super2+super4（双机 TP=16+EP）..."
DENV=(MODEL=$MODEL_CT MODEL_HOST=$ART_DIR_HOST ART_DIR=$ART SERVED=$SERVED
      WORKER_IP=$WI WORKER_HOST=$WH HEAD_IP=$HI HEAD_HOST=$HH)
DEP_OK=0
for attempt in 1 2 3; do
  log "  部署尝试 $attempt/3..."
  env "${DENV[@]}" bash "$DEPLOY" > ${LOG_PREFIX}_deploy.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then DEP_OK=1; log "  部署成功"; break; fi
  log "  部署失败 rc=$rc → 清理重试"
  env "${DENV[@]}" bash "$DEPLOY" --stop > ${LOG_PREFIX}_deploy_stop_$attempt.log 2>&1
  sleep 30
done
[ "$DEP_OK" = 1 ] || { log "ERROR: 部署 3 次均失败（见 ${LOG_PREFIX}_deploy.log）"; exit 1; }

log "5/5 四维评测（同协议；对照 = 底座 3.4507/0.8814/0.9454/0.9512）..."
env PORT=8000 HEAD_IP=$HI MODEL_DIR=$MODEL_CT \
    OUT=/llm_infer_workspace/ylt/glm5.2_deploy/outputs/glm5_w4a8_p2qadreq_4task_hard \
    MARK_DIR=/tmp/glm5_p2qadreq_tier1_markers \
    bash "$EVAL" > ${LOG_PREFIX}_eval.log 2>&1
rc=$?
log "5/5 评测完成 rc=$rc（见 ${LOG_PREFIX}_eval.log；OUT=/llm_infer_workspace/ylt/glm5.2_deploy/outputs/glm5_w4a8_p2qadreq_4task_hard）"
exit $rc
