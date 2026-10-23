#!/bin/bash
# Conservative watchdog for the 5-arm multiquery + HLE-200 (l0/w4a8c8) session.
# Every 10 min: stop zzw containers if pending, restart a component only when
# its process is gone and its DONE marker is absent. Retry cap = 8 per component.
set -uo pipefail
HOSTDIR=/home/yult/llm_infer_workspace
OUT="$HOSTDIR/ylt/glm52-qad/audit1009"
QAD="$HOSTDIR/ylt/npuslim_repo/tools/qad"
DEPLOY_W4A8="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_w4a8_dualnode.sh"
DEPLOY_BF16="$HOSTDIR/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh"
LOG="$OUT/watchdog_1022.log"
W4A8="/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq"
W8="/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq"
MQ_RETRIES=0; HLE_RETRIES=0
log(){ echo "[wd $(date '+%F %T')] $*" | tee -a "$LOG"; }
alive(){ pgrep -f "$1" >/dev/null 2>&1; }

stop_pair(){
  HEAD_IP="$1" WORKER_IP="$2" HEAD_HOST="yult@$1" WORKER_HOST="yult@$2" \
    bash "$DEPLOY_W4A8" --stop >>"$LOG" 2>&1 || true
}
stop_bf16(){ bash "$DEPLOY_BF16" --stop >>"$LOG" 2>&1 || true; }

stop_zzw(){
  local ip c
  for ip in 10.1.0.106 10.1.0.135 10.1.0.57 10.1.0.47; do
    c=$(timeout 15 ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=no "yult@$ip" \
        "docker ps --format '{{.Names}}' | grep '^vllm-ascend-zzw' | head -1" 2>/dev/null || true)
    [ -z "$c" ] && continue
    log "stopping external $c on $ip"
    timeout 90 ssh -o ConnectTimeout=6 -o BatchMode=yes -o StrictHostKeyChecking=no "yult@$ip" \
      "docker exec '$c' bash -lc 'pkill -TERM -f \"[v]llm serve\" || true; pkill -TERM -f \"[V]LLMWorker\" || true; ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1; docker stop -t 60 '$c'" >>"$LOG" 2>&1 || true
  done
}

kill_mq(){
  pkill -f 'bash .*run_multiquery_after_hle.sh' 2>/dev/null || true
  pkill -f 'bash .*chain_multiquery_1019.sh' 2>/dev/null || true
  pkill -f 'bash .*bf16_multiquery_1019.sh' 2>/dev/null || true
}
restart_mq(){
  MQ_RETRIES=$((MQ_RETRIES+1))
  log "RESTART MQ (retry $MQ_RETRIES)"
  kill_mq; stop_pair 10.1.0.57 10.1.0.47; stop_pair 10.1.0.106 10.1.0.135; stop_bf16; sleep 5
  rm -f "$OUT/mq_F.pid" "$OUT/mq_G.pid" "$OUT/mq_bf16.pid"
  setsid nohup bash "$QAD/run_multiquery_after_hle.sh" >>"$LOG" 2>&1 < /dev/null &
}

kill_hle200(){
  pkill -f 'bash .*run_hle200_after_mq.sh' 2>/dev/null || true
  pkill -f 'bash .*chain_hle200_1019.sh' 2>/dev/null || true
}
restart_hle200(){
  HLE_RETRIES=$((HLE_RETRIES+1))
  log "RESTART HLE200 (retry $HLE_RETRIES)"
  kill_hle200; stop_pair 10.1.0.57 10.1.0.47; stop_pair 10.1.0.106 10.1.0.135; sleep 5
  rm -f "$OUT/hle200_F.pid" "$OUT/hle200_G.pid"
  setsid nohup bash "$QAD/run_hle200_after_mq.sh" >>"$LOG" 2>&1 < /dev/null &
}

log "watchdog start"
while true; do
  if [ ! -f "$OUT/MQ_ALL_DONE" ] || [ ! -f "$OUT/HLE200_ALL_DONE" ]; then
    stop_zzw
  fi

  if [ ! -f "$OUT/MQ_ALL_DONE" ]; then
    mq_orch=0; alive 'run_multiquery_after_hle.sh' && mq_orch=1
    f_alive=0; alive 'chain_multiquery_1019.sh F' && f_alive=1
    g_alive=0; alive 'chain_multiquery_1019.sh G' && g_alive=1
    bf_alive=0; alive 'bf16_multiquery_1019.sh' && bf_alive=1
    log "MQ orch=$mq_orch F=$f_alive($([ -f "$OUT/MQ_F_DONE" ] && echo done || echo no)) G=$g_alive($([ -f "$OUT/MQ_G_DONE" ] && echo done || echo no)) bf16=$bf_alive"
    if [ $mq_orch -eq 0 ] && [ $f_alive -eq 0 ] && [ $g_alive -eq 0 ] && [ $bf_alive -eq 0 ]; then
      [ $MQ_RETRIES -ge 8 ] && { log "MQ retry cap reached; watchdog exits"; exit 1; }
      restart_mq
    elif [ ! -f "$OUT/MQ_F_DONE" ] && [ $f_alive -eq 0 ]; then
      [ $MQ_RETRIES -ge 8 ] && { log "MQ retry cap reached; watchdog exits"; exit 1; }
      restart_mq
    elif [ ! -f "$OUT/MQ_G_DONE" ] && [ $g_alive -eq 0 ]; then
      [ $MQ_RETRIES -ge 8 ] && { log "MQ retry cap reached; watchdog exits"; exit 1; }
      restart_mq
    fi
    if [ -f "$OUT/MQ_F_DONE" ] && [ -f "$OUT/MQ_G_DONE" ]; then
      if [ -f "$OUT/mq_bf16_DONE" ]; then
        touch "$OUT/MQ_ALL_DONE"; log "F/G/BF16 all done - marked MQ_ALL_DONE"
      elif [ $bf_alive -eq 0 ]; then
        [ $MQ_RETRIES -ge 8 ] && { log "MQ retry cap reached; watchdog exits"; exit 1; }
        log "F/G done but BF16 absent - restart MQ to run BF16"
        restart_mq
      fi
    fi
  elif [ ! -f "$OUT/HLE200_ALL_DONE" ]; then
    h_orch=0; alive 'run_hle200_after_mq.sh' && h_orch=1
    hf=0; alive 'chain_hle200_1019.sh F' && hf=1
    hg=0; alive 'chain_hle200_1019.sh G' && hg=1
    log "HLE200 orch=$h_orch F=$hf($([ -f "$OUT/HLE200_F_DONE" ] && echo done || echo no)) G=$hg($([ -f "$OUT/HLE200_G_DONE" ] && echo done || echo no))"
    if [ -f "$OUT/HLE200_F_DONE" ] && [ -f "$OUT/HLE200_G_DONE" ]; then
      touch "$OUT/HLE200_ALL_DONE"; log "HLE200 both lanes done - marked ALL_DONE"
    elif [ $h_orch -eq 0 ] && [ $hf -eq 0 ] && [ $hg -eq 0 ]; then
      [ $HLE_RETRIES -ge 8 ] && { log "HLE200 retry cap reached; watchdog exits"; exit 1; }
      restart_hle200
    elif [ ! -f "$OUT/HLE200_F_DONE" ] && [ $hf -eq 0 ]; then
      [ $HLE_RETRIES -ge 8 ] && { log "HLE200 retry cap reached; watchdog exits"; exit 1; }
      restart_hle200
    elif [ ! -f "$OUT/HLE200_G_DONE" ] && [ $hg -eq 0 ]; then
      [ $HLE_RETRIES -ge 8 ] && { log "HLE200 retry cap reached; watchdog exits"; exit 1; }
      restart_hle200
    fi
  else
    log "ALL DONE; watchdog exits"
    exit 0
  fi
  sleep 600
done
