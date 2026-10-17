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

log "=== stop zzw service on .106/.135 (session priority, user-authorized) ==="
for ip in 10.1.0.106 10.1.0.135; do
  ssh_q yult@$ip "docker exec vllm-ascend-zzw-v0.23.0 bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop vllm-ascend-zzw-v0.23.0" >>"$LOG" 2>&1
  log "  $ip: zzw stop issued"
done
sleep 30
log "  .106 free: $(ssh_q yult@10.1.0.106 'npu-smi info 2>/dev/null | grep -c "No running processes"')"

log "=== BF16 deploy (4 nodes TP=32, maxlen 69632, eager, mem 0.93) ==="
MAX_MODEL_LEN=69632 MEM_UTIL=0.93 ENFORCE_EAGER=1 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048 \
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
run_probe ruler2   ruler_hard_probe_v2.py "--length 65536"
run_probe longform longform_panel_probe.py
run_probe safety   safety_probe.py

log "=== BF16 stop ==="
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/T1BF16_DONE"
log "BF16 T1 ARM DONE"
