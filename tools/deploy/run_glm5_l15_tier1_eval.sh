#!/bin/bash
# ===== Phase D 路径参数化（env 带默认回退；不改数值语义）=====
SHARED=${SHARED:-}
SHARED_HOST=${SHARED_HOST:-/home/yult}
# =========================================================
# run_glm5_l15_tier1_eval.sh - GLM-5.2 Layer-1.5 (C 产物) Tier-1 评测
# 16 卡窗口 (super1+super3, e6_noawq_env, TP=16 ray), server 在 head 10.1.0.105:8000。
# Tier-1 = round1 mmlu+gsm8k (5-shot conc=64 maxlen=8192) / round2 wikitext (ppl@1024 conc=2) /
#          round3 humaneval (0-shot conc=8 greedy + code_eval 离线 patch)。
# 执行顺序（0924 起，AGENTS#12 合规）：wikitext → humaneval → mmlu+gsm8k（快判据先行，异常即停省重 round；
#          分值与顺序无关；tag 名保持历史绑定不变）。
# 口径与 run_glm5_p1base_l0_tier1_eval.sh / run_glm52_bf16_eval_c3.sh 逐字一致（四维公平对比）。
# Run ON C3_super1 (head, 容器 --net=host, localhost==head)：
#   ssh C3_super1 "setsid nohup env PORT=8000 MODEL_DIR=<l15 artifact> OUT=<dir> bash .../run_glm5_l15_tier1_eval.sh > .../eval_l15_tier1.log 2>&1 < /dev/null &"
set -uo pipefail

CONTAINER="${CONTAINER:-e6_noawq_env}"
PORT="${PORT:-8000}"
HEAD_IP="${HEAD_IP:-10.1.0.105}"     # 默认 super1；Path 2 用 super5=10.1.0.135
SERVER="http://${HEAD_IP}:${PORT}"
BASE_URL="${SERVER}/v1/completions"
MODEL_DIR="${MODEL_DIR:-/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_modelslim_replica_layer1_5_npu}"
CODEEVAL_DIR="/ylt/qwen3_deploy/qwen3_30b_w4a8"
OUT="${OUT:-/ylt/glm5.2_deploy/outputs/glm5_w4a8_l15_4task_hard}"
WORKDIR="/ylt/glm5.2_deploy"
MARK_DIR="${MARK_DIR:-/tmp/glm5_l15_tier1_markers}"

mkdir -p "$MARK_DIR"
marker() { [ -f "${MARK_DIR}/$1" ]; }
set_marker() { touch "${MARK_DIR}/$1"; }

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${SERVER}/health" 2>/dev/null || echo 000)"
[ "$code" = "200" ] || { echo "ERROR: server not up (${code}) - 先部署 Layer-1.5 产物" >&2; exit 1; }

# --- humaneval code_eval offline patch (idempotent) ---
echo "== patch humaneval code_eval =="
docker exec "${CONTAINER}" bash -lc "
  f=\$(find /usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/humaneval -maxdepth 1 -name utils.py 2>/dev/null | head -1)
  if [ -z \"\$f\" ]; then echo \"humaneval utils not found\"; exit 0; fi
  if grep -q \"qwen3_30b_w4a8/code_eval.py\" \"\$f\"; then echo \"patch already applied\"; exit 0; fi
  cp \"\$f\" \"\${f}.bak_\$(date +%s)\"
  sed -i \"s|evaluate.load(\\\"code_eval\\\")|evaluate.load(\\\"${CODEEVAL_DIR}/code_eval.py\\\")|\" \"\$f\"
  echo \"patch applied to \$f\"
"

run_round() {
  local tag="$1" tasks="$2" fewshot="$3" conc="$4" maxlen="$5" env_extra="$6" lm_extra="${7:-}"
  if marker "$tag"; then echo "=== ${tag} already done - skip ==="; return 0; fi
  echo "=== [$(date '+%F %T')] round ${tag}: tasks=${tasks}, fewshot=${fewshot}, conc=${conc}, maxlen=${maxlen} ==="
  local rc
  docker exec \
    -w "${WORKDIR}" \
    -e HF_HOME=/hf_home \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e OPENAI_API_KEY=EMPTY -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e VLLM_RPC_TIMEOUT=360000 \
    ${env_extra} \
    "${CONTAINER}" bash -lc "PYTHONUNBUFFERED=1 lm_eval \
      --model local-completions \
      --model_args base_url=${BASE_URL},tokenizer=${MODEL_DIR},num_concurrent=${conc},max_length=${maxlen} \
      --tasks ${tasks} \
      --num_fewshot ${fewshot} \
      --gen_kwargs temperature=0 \
      --batch_size auto \
      ${lm_extra} \
      --output_path ${OUT} \
      --log_samples"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! round ${tag} FAILED (exit=$rc)"; return $rc; fi
  set_marker "$tag"
  echo "=== round ${tag} done ==="
}

echo "=== Tier-1 round 1/3: wikitext ppl @1024, conc=2 （AGENTS#12 先行判据）==="
run_round "round2" "wikitext" 0 2 1024 "" "" || exit 1

echo "=== Tier-1 round 2/3: humaneval 0-shot, conc=8, greedy ==="
run_round "round3" "humaneval" 0 8 8192 "-e HF_ALLOW_CODE_EVAL=1" "--confirm_run_unsafe_code" || exit 1

echo "=== Tier-1 round 3/3: mmlu + gsm8k 5-shot, conc=64, maxlen=8192 ==="
run_round "round1" "mmlu,gsm8k" 5 64 8192 "" "" || exit 1

set_marker tier1_done
echo "TIER-1 ALL DONE [$(date '+%F %T')]"
