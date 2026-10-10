#!/bin/bash
# run_glm52_bf16_eval_c3.sh - GLM-5.2 BF16 baseline eval on s1/s2/s3/s4 (10.1.0.x) TP=32 server.
# 口径与 GLM-5.2 W4A8 C3 测评（run_glm52_w4a8_eval_c3.sh）完全一致，保证公平对比：
#   model_args 必须逐字一致（base_url, tokenizer, num_concurrent, max_length）。
#   注意：不可加 tokenized_requests=False（历史脚本虽用 False，但 W4A8 C3 全部 4 轮均为
#   默认 True；且该参数实测对速度零提升，只有公平性风险 -> 2026-09-03 已回滚）。
#   round1: mmlu + gsm8k  5-shot, conc=64, maxlen=8192
#   round2: wikitext ppl @1024  0-shot, conc=2,  maxlen=1024
#   round3: humaneval     0-shot, conc=8,  maxlen=8192, greedy, HF_ALLOW_CODE_EVAL=1, --confirm_run_unsafe_code
#   round4: 7task (arc_challenge,arc_easy,boolq,headqa_en,openbookqa,piqa,winogrande) 0-shot, conc=16, maxlen=8192
# 每轮 marker（s1:/tmp/glm52_bf16_c3_markers/{round1,round2,round3,7task,all_done}）断点续跑。
# 在 s1 上运行（容器 --net=host，localhost == head）：
#   nohup bash run_glm52_bf16_eval_c3.sh > /tmp/eval_glm52_bf16_c3.log 2>&1 &
set -uo pipefail

CONTAINER="glm5_bf16_env"
PORT="${PORT:-9081}"
SERVER="http://localhost:${PORT}"
BASE_URL="${SERVER}/v1/completions"
MODEL_DIR="/llm_infer_workspace/models/GLM-5.2-bf16/snapshots/master"
CODEEVAL_DIR="/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8"
OUT="${OUT:-/llm_infer_workspace/ylt/glm5.2_deploy/outputs/glm5_bf16_4task_hard}"
OUT7="${OUT7:-/llm_infer_workspace/ylt/glm5.2_deploy/outputs/glm5_bf16_7task}"
WORKDIR="/llm_infer_workspace/ylt/glm5.2_deploy"
MARK_DIR="${MARK_DIR:-/tmp/glm52_bf16_c3_markers}"

mkdir -p "$MARK_DIR"

marker() { [ -f "${MARK_DIR}/$1" ]; }
set_marker() { touch "${MARK_DIR}/$1"; }

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${SERVER}/health" 2>/dev/null || echo 000)"
[ "$code" = "200" ] || { echo "ERROR: server not up (${code}) - deploy first" >&2; exit 1; }

# --- humaneval code_eval offline patch (idempotent) ---
# NOTE: 外层必须用双引号（并转义内部 \$），否则 ${CODEEVAL_DIR} 不展开，
#       sed 会把 code_eval 换成 /code_eval.py 导致 FileNotFoundError（2026-09-04 已踩坑修复）。
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
  local tag="$1" tasks="$2" fewshot="$3" conc="$4" maxlen="$5" env_extra="$6" out="$7" lm_extra="${8:-}"
  if marker "$tag"; then echo "=== ${tag} already done - skip ==="; return 0; fi
  echo "=== [$(date '+%F %T')] round ${tag}: tasks=${tasks}, fewshot=${fewshot}, conc=${conc}, maxlen=${maxlen} ==="
  local rc
  docker exec \
    -w "${WORKDIR}" \
    -e HF_HOME=/llm_infer_workspace/hf_home \
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
      --output_path ${out} \
      --log_samples"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! round ${tag} FAILED (exit=$rc)"
    return $rc
  fi
  set_marker "$tag"
  echo "=== round ${tag} done ==="
}

echo "=== round 1/4: mmlu + gsm8k 5-shot, conc=64, maxlen=8192 ==="
run_round "round1" "mmlu,gsm8k" 5 64 8192 "" "$OUT" || exit 1

echo "=== round 2/4: wikitext ppl @1024, conc=2 ==="
run_round "round2" "wikitext" 0 2 1024 "" "$OUT" || exit 1

echo "=== round 3/4: humaneval 0-shot, conc=8, greedy ==="
run_round "round3" "humaneval" 0 8 8192 "-e HF_ALLOW_CODE_EVAL=1" "$OUT" "--confirm_run_unsafe_code" || exit 1

echo "=== round 4/4: 7-task 0-shot, conc=16, maxlen=8192 ==="
run_round "7task" "arc_challenge,arc_easy,boolq,headqa_en,openbookqa,piqa,winogrande" 0 16 8192 "" "$OUT7" || exit 1

set_marker all_done
echo "ALL DONE [$(date '+%F %T')]"
