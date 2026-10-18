#!/bin/bash
# BF16 round: manual continuation after chain takeover (bumped workers; serve stays up).
set -uo pipefail
OUT=/home/yult/llm_infer_workspace/ylt/glm52-qad/audit1009
P2=/home/yult/llm_infer_workspace/ylt/glm52-qad/p2bcd
DEPLOY=/home/yult/llm_infer_workspace/ylt/npuslim_repo/tools/deploy/deploy_glm52_bf16_4node_eval.sh
API="http://10.1.0.106:9081/v1/chat/completions"
MODEL=glm52-bf16-eval
LOG="$OUT/t1bf16.log"
log(){ echo "[t1bf16m $(date '+%F %T')] $*" | tee -a "$LOG"; }
run(){ local tag="$1" probe="$2"; shift 2
  log "=== BF16: $tag ==="
  python3 "$P2/$probe" --api "$API" --model "$MODEL" --out "$OUT/t1bf16_${tag}.jsonl" "$@" >>"$LOG" 2>&1
  log "$tag rc=$?"
}
run simpleqa simpleqa_probe.py --workers 32
run ifbench  ifbench_probe.py --workers 32
run ruler2   ruler_hard_probe_v2.py --length 34000 --workers 4
run longform longform_panel_probe.py --workers 16
run safety   safety_probe.py --workers 32
log "=== BF16: LCB generation (1055) ==="
python3 "$P2/lcb_probe.py" --api "$API" --model "$MODEL" --out "$OUT/lcb_bf16_gen.jsonl" --workers 32 >>"$LOG" 2>&1
log "lcb gen rc=$?"
log "=== BF16: LCB scoring ==="
python3 "$P2/lcb_score.py" --gen "$OUT/lcb_bf16_gen.jsonl" --out "$OUT/lcb_bf16_summary.json" --workers 24 >>"$LOG" 2>&1
log "lcb score rc=$?"
touch "$OUT/lcb_bf16_DONE"
log "=== BF16 stop ==="
bash "$DEPLOY" --stop >>"$LOG" 2>&1
touch "$OUT/T1BF16_DONE"
log "BF16 T1 ARM DONE (manual continuation)"
