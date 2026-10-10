#!/bin/bash
# m5_tier1_noawq_chain_v5.sh — M5 noAWQ 产物 Tier-1 三轮评测（super3 宿主机执行）。
# ★ v4 = 纯 v2（08-28 serve+eval 协议，探针 0.8225 已验证正确）仅改 max-num-batched-tokens 4096→8192 提速尝试。
#   v3 教训（20:49）：v1 风格 serve（无 EP/npugraph）+ 正确 eval 参数 → 探针 0.2625 随机水平——
#   Qwen3-30B W4A8 在 C3 上 logprobs 正确性由 08-28 serve 参数集决定，不能换回快参数。
#   探针门保留：v4 探针 <0.6 则立即回退纯 v2。
# ★ v2 = 严格复刻 08-28 4task_hard 协议（唯一出过健康数值 0.8102/0.8886/11.26/0.7317 的配置）：
#   serve: --enable-expert-parallel + npugraph_ex/fuse_muls_add + FULL_DECODE_ONLY + max-model-len 4096 + mem 0.9
#   eval:  tokenizer_backend=huggingface, tokenized_requests=False, max_length=4096（v1 缺这些 → /tokenizer_info 404 → mmlu/gsm8k 随机水平）
#   round2 wikitext 对齐 08-28（conc=4, max_length=4096）
#   TP: 8→4（坑⑩：C3 硬件 Qwen3-30B W4A8 TP=8 GMM k=96 崩，TP=4=192 ✓）
# 前置探针: mmlu --limit 50 健康门（acc≥0.6 才继续，防止再烧 2h 出随机数）
# v1 记录（失败）: mmlu 0.2568 / gsm8k 0.0053 / round2 EngineCore 崩——v1 配置与 08-28 协议不符所致，非产物问题（产物 in-process raw ppl 7.45 健康）
# 与 m5_tier1_chain.sh 的差异：
#   * PRODUCT = noAWQ 隔离 requant 产物（16:55 完成，跳过 phase1 等待）
#   * CT = e6_noawq_env（super3 当前 UDA 设备持有容器；glm5_qad_env 已停，启动它会与 e6 的登记冲突，坑⑨）
#   * 容器内路径统一 /llm_infer_workspace（e6 容器无 /home/yult/... 挂载，坑⑫）
#   * round2 wikitext max_length 1024→8192（对齐 08-28 4task_hard 全上下文口径，基线 11.26）
#   * phase2 保留 code_eval 离线补丁（09-10 基线 humaneval 全 None 的教训）
# 基线对照（08-28 server 协议，同产物 08-27）：mmlu 0.8102 / gsm8k 0.8886 / wikitext 11.26 / humaneval 0.7317
# 用法: ssh C3_super3 "setsid nohup bash /home/yult/llm_infer_workspace/ylt/glm52-qad/m5_tier1_noawq_chain.sh > /home/yult/llm_infer_workspace/ylt/glm52-qad/m5_tier1_noawq_chain_nohup.log 2>&1 < /dev/null &"
set -uo pipefail

CT="${CT:-e6_noawq_env}"
PORT=8001
SERVED="qwen3-30b-w4a8-baseline"
# 容器内路径（vllm serve / lm_eval 都在容器内跑）
MODEL_DIR="/llm_infer_workspace/ylt/models/qwen3_30b_a3b_2507-w4a8_npu"
CODEEVAL_DIR="/llm_infer_workspace/ylt/qwen3_deploy/qwen3_30b_w4a8"
# 宿主路径（预检用，坑⑫）
MODEL_DIR_HOST="/home/yult/llm_infer_workspace/ylt/models/qwen3_30b_a3b_2507-w4a8_npu"
OUT="/llm_infer_workspace/ylt/glm52-qad/tier1_baseline_w4a8_v5bproto"
OUT_HOST="/home/yult/llm_infer_workspace/ylt/glm52-qad/tier1_baseline_w4a8_v5bproto"
MARK="/home/yult/llm_infer_workspace/ylt/glm52-qad/tier1_baseline_w4a8_v5bproto_markers"
SERVER="http://localhost:${PORT}"
UTILS=/usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/humaneval/utils.py
LOG=/home/yult/llm_infer_workspace/ylt/glm52-qad/tier1_baseline_w4a8_chain_v61.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

mkdir -p "$OUT_HOST" "$MARK"

# ---------- phase 0: 产物在位 ----------
[ -f "${MODEL_DIR_HOST}/quant_model_description.json" ] || { log "PHASE0 产物 description 缺失，中止"; exit 1; }
log "PHASE0 OK: ${MODEL_DIR_HOST}"

# ---------- phase 1: deploy ----------
log "PHASE1 部署 noAWQ 产物（TP=4, ascend quant, 容器 ${CT}）..."
docker exec "$CT" bash -lc \
  "grep -q 'code_eval.py' ${UTILS} 2>/dev/null || sed -i 's|hf_evaluate.load(\"code_eval\")|hf_evaluate.load(\"${CODEEVAL_DIR}/code_eval.py\")|' ${UTILS}" || true
for p in patch_deepseek_v2_loader.py patch_npu_fx_compiler.py; do
  docker exec "$CT" python3 "/llm_infer_workspace/ylt/npuslim/tools/serve/patches/$p" 2>&1 | tail -1 | sed 's/^/  [patch] /' >> "$LOG"
done
docker exec "$CT" bash -lc "pkill -f 'vllm serve.*qwen3_30b_a3b_2507-w4a8_npu' 2>/dev/null; sleep 2" || true  # ★ 只杀基线产物旧 serve，绝不误伤同容器 v5b（noAWQ, 端口8000）
docker exec -d -e ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 -e NPUSLIM_PLUGIN_ENABLE=1 \
  -e OMP_NUM_THREADS=48 -e HF_HOME=/llm_infer_workspace/hf_home \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$CT" bash -lc "nohup vllm serve ${MODEL_DIR} --served-model-name ${SERVED} \
    --tensor-parallel-size 4 --quantization ascend --enable-expert-parallel --dtype auto \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --block-size 128 \
    --max-num-batched-tokens 4096 --max-num-seqs 16 --trust-remote-code --enable-chunked-prefill \
    --additional-config '{\"fuse_muls_add\": true}' \
    --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' \
    --host 0.0.0.0 --port ${PORT} > /llm_infer_workspace/ylt/glm52-qad/vllm_baseline_w4a8_retest_v61.log 2>&1"
log "PHASE1 serve 已拉起，等待 health（最多 40 min）..."
UP=0
for i in $(seq 1 80); do
  sleep 30
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  [ $((i % 4)) -eq 0 ] && log "  [$(($i*30))s] health=$code"
  if [ "$code" = "200" ]; then UP=1; break; fi
done
[ "$UP" = "1" ] || { log "PHASE1 health 超时，查看 /llm_infer_workspace/ylt/glm52-qad/vllm_baseline_w4a8_retest_v61.log"; exit 1; }
log "PHASE1 OK: server up"

# ---------- phase 2: Tier-1 eval ----------
MODEL_ARGS="pretrained=${SERVED},tokenizer=${MODEL_DIR},base_url=${SERVER}/v1/completions,tokenizer_backend=huggingface,tokenized_requests=False,max_length=4096,trust_remote_code=True"

run_round() {
  local tag="$1" tasks="$2" fewshot="$3" conc="$4" extra="$5"
  [ -f "$MARK/$tag" ] && { log "round $tag 已完成（marker）跳过"; return 0; }
  docker exec \
    -w /llm_infer_workspace/ylt/npuslim \
    -e HF_HOME=/llm_infer_workspace/hf_home \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e OPENAI_API_KEY=EMPTY -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e VLLM_RPC_TIMEOUT=360000 ${extra} \
    "$CT" bash -lc "PYTHONUNBUFFERED=1 lm_eval \
      --model local-completions \
      --model_args \"${MODEL_ARGS},num_concurrent=${conc}\" \
      --tasks ${tasks} --num_fewshot ${fewshot} --gen_kwargs temperature=0 \
      --batch_size auto --output_path ${OUT} --log_samples --confirm_run_unsafe_code" \
    >> "$LOG" 2>&1 && { touch "$MARK/$tag"; log "round $tag done"; } || { log "round $tag FAILED"; return 1; }
}

# ---------- phase 2a: 健康门探针 ----------
log "PHASE2a mmlu --limit 50 探针（acc>=0.6 才继续）..."
docker exec \
  -w /llm_infer_workspace/ylt/npuslim \
  -e HF_HOME=/llm_infer_workspace/hf_home \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OPENAI_API_KEY=EMPTY -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  "$CT" bash -lc "PYTHONUNBUFFERED=1 lm_eval \
    --model local-completions \
    --model_args \"${MODEL_ARGS},num_concurrent=64\" \
    --tasks mmlu --num_fewshot 5 --limit 50 --gen_kwargs temperature=0 \
    --batch_size auto --output_path ${OUT}_probe" >> "$LOG" 2>&1
ACC="$(python3 - "$OUT_HOST"_probe <<'PYEOF2'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True)
if not fs:
    print("NO_RESULT")
else:
    d = json.load(open(sorted(fs)[-1]))
    r = d["results"].get("mmlu", {})
    print(r.get("acc,none", "NO_ACC"))
PYEOF2
)"
log "  probe mmlu acc(50) = ${ACC}"
case "$ACC" in
  NO_RESULT|NO_ACC|'') log "PROBE PARSE FAIL - 中止"; exit 1 ;;
esac
awk -v a="$ACC" 'BEGIN{exit !(a+0>=0.6)}' || { log "PROBE FAILED (acc=${ACC} < 0.6) - 协议仍有问题，中止"; exit 1; }
log "PHASE2a PROBE OK"

log "PHASE2 round1: mmlu+gsm8k 5shot conc64（08-28 同参）"
run_round round1 "mmlu,gsm8k" 5 64 "" || exit 1
log "PHASE2 round2: wikitext conc4 max_length4096（08-28 同参，基线 11.26）"
run_round round2 "wikitext" 0 4 "" || exit 1
log "PHASE2 round3: humaneval greedy conc16 + code_eval 离线判分（08-28 同参）"
run_round round3 "humaneval" 0 16 "-e HF_ALLOW_CODE_EVAL=1" || exit 1

touch "$MARK/ALL_DONE"
log "基线 W4A8 同协议复测 CHAIN v6.1 ALL DONE — 结果: ${OUT_HOST}；对照 08-28 原基线（C3f+npugraph）: mmlu 0.8102 / gsm8k 0.8886 / wikitext 11.26 / humaneval 0.7317；M5 noAWQ（v5b 同协议 C3）: round1 mmlu 0.8111 / gsm8k 0.8704（v4，待 v5b 交叉验证）"
