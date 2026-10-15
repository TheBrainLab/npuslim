#!/bin/bash
# =============================================================================
# run_l0_instruments_1015.sh — run the L0 fidelity instruments + PPL–length
# curve against the currently deployed W4A8 server on C3_super3 (10.1.0.57).
# Results land in glm52-qad/audit1009/l0_fidelity_1015/.
#
# env: NODE_HOST CONT MODEL_NAME ARTIFACT TAG
# =============================================================================
set -uo pipefail
NODE_HOST="${NODE_HOST:-10.1.0.57}"
CONT="${CONT:-e6_noawq_env}"
MODEL_NAME="${MODEL_NAME:-glm52-w4a8-eval}"
ARTIFACT="${ARTIFACT:-/llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni}"
TAG="${TAG:-l0}"
SHARED=/llm_infer_workspace
OUT="${OUT_DIR:-$SHARED/ylt/glm52-qad/audit1009/l0_fidelity_1015}"
CACHE="$SHARED/ylt/glm52-qad/precompute/logits_cache"
TOOLS="$SHARED/ylt/npuslim_repo/tools"
API="${API:-http://127.0.0.1:8000/v1/completions}"
LENGTHS_VAL="${LENGTHS_VAL:-1024,2048,4096,16384,65536,131072}"
LENGTHS_TRAIN="${LENGTHS_TRAIN:-4096,16384,65536,131072,262140}"
WT_DIR="$SHARED/ylt/npuslim/workspace/wikitext-2-raw-v1"

sshq(){ timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@" 2>/dev/null; }

sshq "$NODE_HOST" "mkdir -p $OUT"
chmod +x "$TOOLS/deploy/eval_ppl_length.py" "$TOOLS/qad/eval_dist_fidelity.py" 2>/dev/null || true

run_detached() {  # $1=logfile on host, rest=container command
  local log="$1"; shift
  sshq "$NODE_HOST" "docker exec -d $CONT bash -lc '$* > $OUT/$log 2>&1'"
  echo "  -> $OUT/$log"
}

echo "== [1/3] distribution fidelity (teacher cache) =="
run_detached "dist_fidelity_${TAG}.log" \
  "python3 $TOOLS/qad/eval_dist_fidelity.py --cache $CACHE --api $API --model $MODEL_NAME --tokenizer $ARTIFACT --n-shards 6 --per-shard 30 --min-rows 256 --shard-stride 1 --max-len 2048 --topk 20 --out $OUT/dist_fidelity_${TAG}.json"

echo "== [2/3] PPL-length (validation) =="
run_detached "ppl_length_val_${TAG}.log" \
  "python3 $TOOLS/deploy/eval_ppl_length.py --api $API --model $MODEL_NAME --tokenizer $ARTIFACT --pages-json $SHARED/ylt/glm5.2_deploy/wt2_validation.json --lengths $LENGTHS_VAL --concurrency 1 --profile --out $OUT/ppl_length_val_${TAG}.json"

echo "== [3/3] PPL-length (train, incl 256K) =="
run_detached "ppl_length_train_${TAG}.log" \
  "python3 $TOOLS/deploy/eval_ppl_length.py --api $API --model $MODEL_NAME --tokenizer $ARTIFACT --parquet $WT_DIR/wikitext-2-raw-v1-train.parquet --lengths $LENGTHS_TRAIN --concurrency 1 --profile --out $OUT/ppl_length_train_${TAG}.json"

echo "all instruments launched; monitor $OUT/*.log"
