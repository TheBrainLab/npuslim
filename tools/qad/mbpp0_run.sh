#!/bin/bash
# Container-side helper: run the 0-prefix mbpp baseline against the local server.
# Protocol is IDENTICAL to mbpp_longctx (P2a) — the only difference is the prompt
# (no 32K prefix), so the 0-prefix vs 32K comparison is protocol-clean.
# Args: MODEL_DIR PORT CONC MAXLEN OUT_C
set -uo pipefail
MODEL_DIR="$1"; PORT="$2"; CONC="$3"; MAXLEN="$4"; OUT_C="$5"
TASK_DIR=/llm_infer_workspace/ylt/glm52-qad/mbpp0
MAXGEN="${MAXGEN:-160}"
export HF_HOME=/llm_infer_workspace/hf_home
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY=EMPTY TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_RPC_TIMEOUT=360000 HF_ALLOW_CODE_EVAL=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
cd "$TASK_DIR"
PYTHONUNBUFFERED=1 lm_eval \
  --model local-completions \
  --model_args "base_url=http://localhost:${PORT}/v1/completions,tokenizer=${MODEL_DIR},num_concurrent=${CONC},max_length=${MAXLEN}" \
  --tasks mbpp0 --include_path "$TASK_DIR" \
  --num_fewshot 0 --gen_kwargs "temperature=0,max_gen_toks=${MAXGEN}" --batch_size auto \
  --confirm_run_unsafe_code --output_path "$OUT_C" --log_samples
