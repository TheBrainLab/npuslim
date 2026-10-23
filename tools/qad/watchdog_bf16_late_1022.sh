#!/bin/bash
# Short watchdog for the late BF16 multiquery completion (4 nodes).
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
LOG="$OUT/watchdog_bf16_late.log"
log(){ echo "[wdbf16 $(date '+%F %T')] $*" | tee -a "$LOG"; }
RETRIES=0
alive(){ pgrep -f "$1" >/dev/null 2>&1; }
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
log "BF16 late watchdog start"
while true; do
  if [ -f "$OUT/mq_bf16_DONE" ]; then log "BF16 late DONE"; exit 0; fi
  stop_zzw
  if alive 'bf16_multiquery_1019.sh'; then
    log "BF16 late alive"
  else
    if [ $RETRIES -ge 2 ]; then log "BF16 late retry cap reached; marking failed"; touch "$OUT/mq_bf16_LATE_FAILED"; exit 1; fi
    RETRIES=$((RETRIES+1)); log "BF16 late dead; restart $RETRIES"
    touch "$OUT/mq_bf16_SKIP" 2>/dev/null || true
    setsid nohup bash "$QAD/bf16_multiquery_1019.sh" >>"$LOG" 2>&1 < /dev/null &
  fi
  sleep 300
done
