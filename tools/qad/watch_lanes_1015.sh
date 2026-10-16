#!/bin/bash
# =============================================================================
# watch_lanes_1015.sh — watchdog for the P1+P2a lanes.
#   Every 5 min: if a lane is not DONE and its driver process is dead, relaunch
#   the lane with the REMAINING products (skip ones that already have both
#   router_<tag>.npz and lctx_<tag>/). Also flags lanes whose log has not grown
#   for >45 min while health != 200 (stuck deploy) by killing+relaunching.
#
#   Lane definitions are inlined (kept in sync with the launch commands).
# =============================================================================
set -uo pipefail
OUT=/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009
HOSTDIR=/home/yult/llm_infer_workspace
CHAIN="$HOSTDIR/ylt/npuslim_repo/tools/qad/chain_lane_1015.sh"
W4A8=/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq
WLOG="$OUT/watch_lanes.log"
log(){ echo "[watch $(date '+%F %T')] $*" | tee -a "$WLOG"; }

lane_nodes(){ case "$1" in
  A) echo "10.1.0.57 10.1.0.47 yult@10.1.0.57 yult@10.1.0.47";;
  B) echo "10.1.0.106 10.1.0.135 yult@10.1.0.106 yult@10.1.0.135";;
esac; }
lane_products(){ case "$1" in
  A) echo "l0|$W4A8/glm5-w4a8_replica_l0_clip_lo05_omni|0.85 w4a8c8|/llm_infer_workspace/models/GLM-5.2-w4a8c8|0.85";;
  B) echo "p2clip|$W4A8/glm5-w4a8_modelslim_replica_layer1_5_path2_clip_seed1234|0.85 w8a8|/llm_infer_workspace/ylt/npuslim_repo/outputs_npu/glm5/gptq/glm5-w8a8_replica_p2clipfold_w8expert|0.92";;
esac; }
product_done(){ local tag="$1"
  # the chain writes router_<tag>.pt (custom hook) and lctx_<tag>/ ; do NOT require .npz
  [ -f "$OUT/router_dump/router_${tag}.pt" ] && [ -d "$OUT/lctx_${tag}" ]; }
lane_alive(){ ps -eo cmd | grep -q "chain_lane_1015.sh" && return 0; return 1; }
lm_eval_active(){ local L="$1" H
  case "$L" in A) H=yult@10.1.0.57;; B) H=yult@10.1.0.106;; esac
  timeout 30 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$H" \
    "docker exec e6_noawq_env bash -lc 'pgrep -cf \"[l]m_eval\"'" 2>/dev/null | grep -qv '^0$'; }

for round in $(seq 1 200); do
  a=$([ -f "$OUT/LANE_A_DONE" ] && echo 1 || echo 0)
  b=$([ -f "$OUT/LANE_B_DONE" ] && echo 1 || echo 0)
  [ "$a" = "1" ] && [ "$b" = "1" ] && { log "both lanes DONE"; break; }

  for L in A B; do
    [ -f "$OUT/LANE_${L}_DONE" ] && continue
    # remaining products
    rem=""
    for item in $(lane_products "$L"); do
      tag="${item%%|*}"
      product_done "$tag" || rem="$rem $item"
    done
    rem="${rem# }"
    if [ -z "$rem" ]; then log "lane $L: all products done but no marker - touch"; touch "$OUT/LANE_${L}_DONE"; continue; fi
    # driver alive? (lane tag is argv[1] so it appears in the cmdline)
    if pgrep -f "chain_lane_1015.sh $L" >/dev/null 2>&1; then
      # log growth check
      sz=$(stat -c %s "$OUT/lane_${L}.log" 2>/dev/null || echo 0)
      prev=$(cat "$OUT/.lane_${L}_sz" 2>/dev/null || echo 0)
      echo "$sz" > "$OUT/.lane_${L}_sz"
      if [ "$sz" = "$prev" ]; then
        stale=$(cat "$OUT/.lane_${L}_stale" 2>/dev/null || echo 0); stale=$((stale+1)); echo "$stale" > "$OUT/.lane_${L}_stale"
        # a running lm_eval legitimately produces no log output for 45+ min -> not stuck
        if lm_eval_active "$L"; then stale=0; echo 0 > "$OUT/.lane_${L}_stale"; fi
        if [ "$stale" -ge 18 ]; then   # ~90 min no growth AND no lm_eval running
          log "lane $L STUCK (no log growth ~90min, no lm_eval) - killing + relaunching with:$rem"
          pkill -f "chain_lane_1015.sh $L" 2>/dev/null
          sleep 5
          set -- $(lane_nodes "$L")
          LANE=$L HEAD_IP=$1 WORKER_IP=$2 HEAD_HOST=$3 WORKER_HOST=$4 PRODUCTS="$rem" \
            setsid nohup bash "$CHAIN" "$L" >>"$OUT/lane_${L}.log" 2>&1 < /dev/null &
          echo 0 > "$OUT/.lane_${L}_stale"
        fi
      else
        echo 0 > "$OUT/.lane_${L}_stale"
      fi
      continue
    fi
    # driver dead -> relaunch remaining
    log "lane $L driver DEAD - relaunching with:$rem"
    set -- $(lane_nodes "$L")
    LANE=$L HEAD_IP=$1 WORKER_IP=$2 HEAD_HOST=$3 WORKER_HOST=$4 PRODUCTS="$rem" \
      setsid nohup bash "$CHAIN" "$L" >>"$OUT/lane_${L}.log" 2>&1 < /dev/null &
  done
  sleep 300
done
log "watchdog exit"
