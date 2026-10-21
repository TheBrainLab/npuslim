#!/bin/bash
# Watchdog v6 (10-21): BF16_re -> parallel (F: l0@65K GPQA upper bound) + (G: w8a8 GPQA high+32K)
set -uo pipefail
cd /home/yult/llm_infer_workspace/ylt
OUT=glm52-qad/audit1009
REPO=npuslim_repo/tools/qad
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
W8=/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq
WLOG="$OUT/watch_night6.log"
log(){ echo "[wnight6 $(date '+%F %T')] $*" | tee -a "$WLOG"; }
log "start (pid $$)"
for it in $(seq 1 144); do
  # F: l0@65K upper bound (after BF16)
  if [ -f "$OUT/aimefix_bf16_re_DONE" ] && [ ! -f "$OUT/gpqa_l0_re_DONE" ] && ! pgrep -f 'chain_gpqa_1019.sh F$' >/dev/null 2>&1; then
    log "l0@65K (F) driver gone -> relaunch"
    env LANE=F CFGSUF=_re HEAD_IP=10.1.0.57 WORKER_IP=10.1.0.47 HEAD_HOST=yult@10.1.0.57 WORKER_HOST=yult@10.1.0.47 \
      PRODUCTS="l0|$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni|0.85|131072" \
      setsid nohup bash "$REPO/chain_gpqa_1019.sh" F >> "$OUT/gpqa_l0_65k.log" 2>&1 < /dev/null &
  fi
  # G: w8a8 GPQA high+32K (after BF16)
  if [ -f "$OUT/aimefix_bf16_re_DONE" ] && [ ! -f "$OUT/gpqa_w8a8_gp32_DONE" ] && ! pgrep -f 'chain_gpqa_1019.sh G$' >/dev/null 2>&1; then
    log "w8a8 gp32 (G) driver gone -> relaunch (stop zzw first)"
    for ip in 10.1.0.106 10.1.0.135; do
      ssh -o BatchMode=yes -o StrictHostKeyChecking=no yult@$ip "for c in \$(docker ps --format '{{.Names}}' | grep zzw); do docker exec \$c bash -lc 'ray stop --force' >/dev/null 2>&1; docker stop \$c >/dev/null 2>&1; done" >/dev/null 2>&1
    done
    env LANE=G CFGSUF=_gp32 HEAD_IP=10.1.0.106 WORKER_IP=10.1.0.135 HEAD_HOST=yult@10.1.0.106 WORKER_HOST=yult@10.1.0.135 \
      PRODUCTS="w8a8|$W8/glm5-w8a8_replica_p2clipfold_w8expert|0.95|69632" \
      setsid nohup bash "$REPO/chain_gpqa_1019.sh" G >> "$OUT/gpqa_w8a8_gp32.log" 2>&1 < /dev/null &
  fi
  # BF16_re: as soon as the two product lanes are done
  if [ -f "$OUT/gpqa_w4a8c8_gp32_DONE" ] && [ -f "$OUT/gpqa_p2clip_gp32_DONE" ] && [ ! -f "$OUT/aimefix_bf16_re_DONE" ]; then
    if ! pgrep -f 'chain_aimefix_bf16_1019.sh$' >/dev/null 2>&1; then
      log "BF16_re driver gone -> relaunch"
      setsid nohup bash "$REPO/chain_aimefix_bf16_1019.sh" >> "$OUT/aimefix_bf16_re.log" 2>&1 < /dev/null &
    fi
  fi
  if [ -f "$OUT/gpqa_l0_re_DONE" ] && [ -f "$OUT/gpqa_w8a8_gp32_DONE" ] && [ -f "$OUT/aimefix_bf16_re_DONE" ]; then
    log "ALL TARGETS DONE - exit"; exit 0
  fi
  sleep 600
done
log "watchdog max iterations"
