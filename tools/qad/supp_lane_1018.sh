#!/bin/bash
# 1018 audit supplements: trace-instructed tracking / cap audits / efficiency.
# PRODUCTS spec: "tag|model|mem|probes"  (probes: comma list of v3c,capaudit,ifbench,eff)
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
DEPLOY="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
P2="$HOSTDIR/ylt/glm52-qad/p2bcd"
HEAD_IP="${HEAD_IP:?}"; WORKER_IP="${WORKER_IP:?}"
HEAD_HOST="${HEAD_HOST:-yult@${HEAD_IP}}"; WORKER_HOST="${WORKER_HOST:-yult@${WORKER_IP}}"
PORT=8000
PRODUCTS="${PRODUCTS:?need PRODUCTS}"
LANE="${LANE:-A}"
LOG="$OUT/supp_${LANE}.log"
log(){ echo "[supp$LANE $(date '+%F %T')] $*" | tee -a "$LOG"; }
run_product(){
  local tag="$1" model="$2" mem="$3" probes="$4"
  if [ -f "$OUT/supp_${tag}_DONE" ]; then log "$tag: already done - skip"; return 0; fi
  log "=== $tag: deploy (mem $mem) ==="
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; sleep 8
  MODEL="$model" SERVED="supp_${tag}" ROUTER_DUMP="" ENFORCE_EAGER=0 \
    HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
    MAX_MODEL_LEN=69632 MEM_UTIL="$mem" MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=16 \
    bash "$DEPLOY" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then log "$tag DEPLOY FAILED"; HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1; return 1; fi
  local api="http://${HEAD_IP}:${PORT}/v1/chat/completions"
  IFS=',' read -ra PL <<< "$probes"
  for p in "${PL[@]}"; do
    case "$p" in
      v3c)      log "=== $tag: v3c (trace-instructed, 9 cases) ==="
                python3 "$P2/ruler_hard_probe_v3c.py" --api "$api" --model "supp_${tag}" --out "$OUT/supp_${tag}_v3c.jsonl" >>"$LOG" 2>&1; log "v3c rc=$?" ;;
      capaudit) log "=== $tag: cap audit simpleqa (30, cap 1024) ==="
                python3 "$P2/cap_audit_simpleqa.py" --api "$api" --model "supp_${tag}" --out "$OUT/supp_${tag}_capaudit.json" --recorded "$OUT/t1b_${tag}_simpleqa.jsonl" >>"$LOG" 2>&1; log "capaudit rc=$?" ;;
      ifbench)  log "=== $tag: ifbench audit (15, cap 1024) ==="
                python3 "$P2/ifbench_probe_audit.py" --api "$api" --model "supp_${tag}" --out "$OUT/supp_${tag}_ifbench_audit.jsonl" --limit 15 >>"$LOG" 2>&1; log "ifbench rc=$?" ;;
      eff)      log "=== $tag: efficiency ==="
                python3 "$P2/efficiency_probe.py" --api "$api" --model "supp_${tag}" --out "$OUT/supp_${tag}_efficiency.json" >>"$LOG" 2>&1; log "eff rc=$?" ;;
    esac
  done
  HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP HEAD_HOST=$HEAD_HOST WORKER_HOST=$WORKER_HOST bash "$DEPLOY" --stop >>"$LOG" 2>&1
  touch "$OUT/supp_${tag}_DONE"
  log "=== $tag DONE ==="
}
for item in $PRODUCTS; do
  tag="${item%%|*}"; rest="${item#*|}"; model="${rest%%|*}"; rest="${rest#*|}"; mem="${rest%%|*}"; probes="${rest#*|}"
  [ "$probes" = "$mem" ] && probes="eff"
  run_product "$tag" "$model" "$mem" "$probes"
done
touch "$OUT/SUPP_${LANE}_DONE"
log "SUPP LANE $LANE DONE"
