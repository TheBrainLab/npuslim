#!/bin/bash
# BF16 reference arm (1017): floor calibration for the 1017 Tier1 axes.
# 6 probes: gmmlu / simpleqa / ifbench / ruler-v2 (64K) / longform / safety.
# Runs on the 4 fixed nodes (.106 head = this control host, TP=32 eager).
# Frees .106/.135 first by stopping zzw's service (user-authorized session priority).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT=$HOSTDIR/ylt/glm52-qad/audit1009
P2=$HOSTDIR/ylt/glm52-qad/p2bcd
DEPLOY=$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
PORT=9081
API="http://10.1.0.106:${PORT}/v1/chat/completions"
MODEL=glm52-bf16-eval
LOG="$OUT/t1bf16.log"
log(){ echo "[t1bf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
ssh_q(){ timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no "$@" 2>/dev/null; }

log "=== stop zzw service on all 4 nodes (session priority, user-authorized) ==="
for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
  ssh_q yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c; done" >>"$LOG" 2>&1
  log "  $ip: zzw stop issued"
done
sleep 30
log "  .106 free: $(ssh_q yult@10.1.0.106 'npu-smi info 2>/dev/null | grep -c "No running processes"')"

log "=== BF16 deploy (4 nodes TP=32, maxlen 36864, eager, mem 0.95) ==="
MAX_MODEL_LEN=36864 MEM_UTIL=0.95 ENFORCE_EAGER=1 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048 \
  bash "$DEPLOY" >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "BF16 DEPLOY FAILED rc=$rc"; exit 1; fi
log "BF16 deploy OK"

run_probe(){ # tag probe extra
  local tag="$1" probe="$2" extra="${3:-}"
  log "=== BF16: $tag ==="
  python3 "$P2/$probe" --api "$API" --model "$MODEL" --out "$OUT/t1bf16_${tag}.jsonl" $extra >>"$LOG" 2>&1
  log "$tag rc=$?"
}
run_probe gmmlu    gmmlu_probe.py
run_probe simpleqa simpleqa_probe.py
run_probe ifbench  ifbench_probe.py
run_probe ruler2   ruler_hard_probe_v2.py "--length 34000"  # 34K floor indicator (64K infeasible at TP=32)
run_probe longform longform_panel_probe.py
run_probe safety   safety_probe.py

# ---- LCB generation (v6, 1055) — needs the full dataset; wait up to 40 min ----
LCBD=$HOSTDIR/ylt/glm52-qad/p2bcd_data/lcb
lcb_ok=0
for i in $(seq 1 8); do
  lcb_ok=1
  for f in test test2 test3 test4 test5 test6; do [ -s "$LCBD/${f}.jsonl" ] || lcb_ok=0; done
  [ "$lcb_ok" = 1 ] && break
  log "  waiting for LCB dataset ($i/8)..."
  sleep 300
done
if [ "$lcb_ok" = 1 ]; then
  log "=== BF16: LCB generation (1055 problems) ==="
  python3 "$P2/lcb_probe.py" --api "$API" --model "$MODEL" --out "$OUT/lcb_bf16_gen.jsonl" --workers 8 >>"$LOG" 2>&1
  log "lcb gen rc=$?"
  log "=== BF16: LCB scoring ==="
  python3 "$P2/lcb_score.py" --gen "$OUT/lcb_bf16_gen.jsonl" --out "$OUT/lcb_bf16_summary.json" --workers 24 >>"$LOG" 2>&1
  log "lcb score rc=$?"
  touch "$OUT/lcb_bf16_DONE"
else
  log "LCB dataset incomplete after 40 min - bf16 LCB SKIPPED"
  touch "$OUT/lcb_bf16_SKIPPED"
fi

log "=== BF16 stop ==="
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/T1BF16_DONE"
log "BF16 T1 ARM DONE"
