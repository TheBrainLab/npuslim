#!/bin/bash
# =============================================================================
# chain_bf16_arms_1016.sh — BF16 reference arms on our 4 nodes (TP=32).
#
#   arm1: router probe (eager + NPUSLIM_ROUTER_DUMP, 4K ctx)
#         -> router_dump/router_bf16.pt   (P1 的 BF16 仪器臂 = 路由轴地板)
#   arm2: P2a 32K mbpp (non-eager, 36K ctx, max_num_seqs=32)
#         -> lctx_bf16/                    (代码轴的 BF16 参照臂)
#
# Idempotent: an arm is skipped when its artifact already exists.
# env: LANE (default BF)
# =============================================================================
set -uo pipefail
LANE="${1:-${LANE:-BF}}"
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx
QAD_C=/llm_infer_workspace/ylt/npuslim_repo/tools/qad
HEAD_HOST=yult@10.1.0.106
CONTAINER=glm5_bf16_env
PORT=9081
MODEL=/llm_infer_workspace/models/GLM-5.2-bf16/snapshots/master
PARQUET=/llm_infer_workspace/ylt/npuslim/workspace/wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet
LOG="$OUT/bf16_arms_${LANE}.log"
log(){ echo "[bf16$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_long(){ timeout 7200 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

stop_all(){
  for ip in 10.1.0.106 10.1.0.57 10.1.0.47 10.1.0.135; do
    ssh_long "yult@${ip}" "docker exec ${CONTAINER} bash -lc 'pkill -9 -f \"[v]llm serve\"; ray stop --force >/dev/null 2>&1; pkill -9 -f \"[r]ay::\"' " >/dev/null 2>&1 || true
  done
}

# ---------------- arm 1: router probe (eager) ----------------
if [ ! -f "$OUT/router_dump/router_bf16.pt" ]; then
  log "=== arm1: BF16 deploy (eager, router dump, 4K ctx) ==="
  ROUTER_DUMP="/llm_infer_workspace/ylt/glm52-qad/audit1009/router_dump/router_bf16.pt" ROUTER_CALLS=1 \
    MAX_MODEL_LEN=4096 MEM_UTIL=0.90 ENFORCE_EAGER=1 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=8 \
    bash "$DEPLOY" >>"$LOG" 2>&1 && log "arm1 deploy OK" || log "arm1 DEPLOY FAILED"
  log "=== arm1: router probe (fixed wikitext 2048-token prompt) ==="
  ssh_long "$HEAD_HOST" "docker exec $CONTAINER python3 $QAD_C/router_probe_patch.py --api http://127.0.0.1:${PORT}/v1/completions --model glm52-bf16-eval --tokenizer $MODEL --parquet $PARQUET --tokens 2048" >>"$LOG" 2>&1
  log "arm1 probe rc=$?"
  ls -la "$OUT/router_dump/router_bf16.pt" 2>/dev/null | tail -1 >>"$LOG"
  stop_all
fi

# ---------------- arm 2: P2a 32K mbpp (non-eager) ----------------
if [ ! -d "$OUT/lctx_bf16" ]; then
  log "=== arm2: BF16 deploy (non-eager, 36K ctx) ==="
  ROUTER_DUMP="" MAX_MODEL_LEN=36864 MEM_UTIL=0.92 ENFORCE_EAGER=0 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32 \
    bash "$DEPLOY" >>"$LOG" 2>&1 && log "arm2 deploy OK" || log "arm2 DEPLOY FAILED"
  log "=== arm2: metric preflight ==="
  if HEAD_HOST="$HEAD_HOST" CONTAINER="$CONTAINER" TASK_DIR="$TASK_DIR" bash "$QAD_C/preflight_metric_1016.sh" >>"$LOG" 2>&1; then
    log "preflight OK (metric self-test = 1.0)"
  else
    log "arm2 PREFLIGHT FAILED (metric broken) - see log"
  fi
  log "=== arm2: lm_eval mbpp_longctx (32K prefix) ==="
  ssh_long "$HEAD_HOST" "docker exec -w $TASK_DIR $CONTAINER bash $TASK_DIR/run_lm_eval.sh $MODEL $PORT 6 36864 /llm_infer_workspace/ylt/glm52-qad/audit1009/lctx_bf16" >>"$LOG" 2>&1
  log "lm_eval rc=$?"
  stop_all
fi

touch "$OUT/BF16_${LANE}_DONE"
log "BF16 ARMS $LANE DONE"
