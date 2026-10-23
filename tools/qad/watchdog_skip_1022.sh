#!/bin/bash
# Skip-aware watchdog: at hard cutoffs, mark a persistently failing task SKIP
# and let the next stage continue. Does not retry indefinitely.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
DEPLOY_W4A8="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
DEPLOY_BF16="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
LOG="$OUT/watchdog_skip_1022.log"
log(){ echo "[wdskip $(date '+%F %T')] $*" | tee -a "$LOG"; }
DL_FG=$(date -d '2026-10-23 00:30:00' +%s)
DL_BF16=$(date -d '2026-10-23 01:30:00' +%s)
DL_HLE=$(date -d '2026-10-23 12:00:00' +%s)

stop_pair(){ HEAD_IP="$1" WORKER_IP="$2" HEAD_HOST="yult@$1" WORKER_HOST="yult@$2" bash "$DEPLOY_W4A8" --stop >>"$LOG" 2>&1 || true; }
stop_bf16(){ bash "$DEPLOY_BF16" --stop >>"$LOG" 2>&1 || true; }
stop_zzw(){
  local ip c
  for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
    c=$(timeout 12 ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "yult@$ip" \
        "docker ps --format '{{.Names}}' | grep '^vllm-ascend-zzw' | head -1" 2>/dev/null || true)
    [ -z "$c" ] && continue
    log "stopping external $c on $ip"
    timeout 80 ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "yult@$ip" \
      "docker exec '$c' bash -lc 'pkill -TERM -f \"[v]llm serve\" || true; pkill -TERM -f \"[V]LLMWorker\" || true; ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1; docker stop -t 60 '$c'" >>"$LOG" 2>&1 || true
  done
}
skip_f(){
  log "F/G cutoff: skip w4a8c8 (if missing)"
  pkill -f 'bash .*chain_multiquery_1019.sh F' 2>/dev/null || true
  stop_pair 10.1.0.57 10.1.0.47
  touch "$OUT/mq_w4a8c8_SKIP" "$OUT/MQ_F_DONE"
}
skip_g(){
  log "F/G cutoff: skip w8a8 (if missing)"
  pkill -f 'bash .*chain_multiquery_w8a8_64k_1019.sh' 2>/dev/null || true
  pkill -f 'bash .*chain_multiquery_1019.sh G' 2>/dev/null || true
  stop_pair 10.1.0.106 10.1.0.135
  touch "$OUT/mq_w8a8_SKIP" "$OUT/MQ_G_DONE"
}
skip_bf16(){
  log "BF16 cutoff: skip BF16"
  pkill -f 'bash .*bf16_multiquery_1019.sh' 2>/dev/null || true
  stop_bf16
  touch "$OUT/mq_bf16_SKIP" "$OUT/MQ_ALL_DONE"
}
skip_hle(){
  log "HLE200 cutoff: skip remaining HLE200 lanes"
  pkill -f 'bash .*run_hle200_after_mq.sh' 2>/dev/null || true
  pkill -f 'bash .*chain_hle200_1019.sh' 2>/dev/null || true
  stop_pair 10.1.0.57 10.1.0.47; stop_pair 10.1.0.106 10.1.0.135
  touch "$OUT/HLE200_SKIP" "$OUT/HLE200_ALL_DONE"
}

log "skip-aware watchdog start; cutoffs FG=00:30 BF16=01:30 HLE=12:00"
while true; do
  now=$(date +%s)
  if [ ! -f "$OUT/MQ_ALL_DONE" ] || [ ! -f "$OUT/HLE200_ALL_DONE" ]; then stop_zzw; fi
  if [ "$now" -ge "$DL_FG" ]; then
    [ -f "$OUT/MQ_F_DONE" ] || skip_f
    [ -f "$OUT/MQ_G_DONE" ] || skip_g
  fi
  if [ "$now" -ge "$DL_BF16" ] && [ ! -f "$OUT/MQ_ALL_DONE" ]; then skip_bf16; fi
  if [ "$now" -ge "$DL_HLE" ] && [ ! -f "$OUT/HLE200_ALL_DONE" ]; then skip_hle; fi
  log "status MQ_F=$([ -f "$OUT/MQ_F_DONE" ] && echo done || echo no) MQ_G=$([ -f "$OUT/MQ_G_DONE" ] && echo done || echo no) BF16=$([ -f "$OUT/mq_bf16_DONE" ] && echo done || echo no) MQ_ALL=$([ -f "$OUT/MQ_ALL_DONE" ] && echo done || echo no) HLE200=$([ -f "$OUT/HLE200_ALL_DONE" ] && echo done || echo no)"
  if [ -f "$OUT/MQ_ALL_DONE" ] && [ -f "$OUT/HLE200_ALL_DONE" ]; then log "all done; watchdog exits"; exit 0; fi
  sleep 600
done
