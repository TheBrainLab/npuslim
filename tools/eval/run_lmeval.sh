#!/bin/bash

# ==============================================================================
# LM-Evaluation-Harness Wrapper Script
# ==============================================================================
# Usage: bash run_lmeval.sh [MODEL_PATH] [OPTIONS]
# ==============================================================================

# Resolve paths independent of working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/common.sh"
PROJECT_ROOT=$(get_project_root)

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Run lm-evaluation-harness benchmarks via OpenAI-compatible API.
Requires a running vLLM server (use tools/serve/deploy_vllm.sh first).

Arguments:
  MODEL_PATH               Path or name of model for reference (required)

Options:
  --model-name NAME          Model name sent to API (default: derived from MODEL_PATH)
  --url URL                  API endpoint URL (default: http://127.0.0.1:\$PORT/v1/completions)
  --port PORT                Server port for auto URL (default: 8080)
  --tasks LIST               Comma-separated tasks (default: wikitext)
  --fewshot N                Number of few-shot examples (default: 0)
  --batch-size SIZE          Batch size or 'auto' (default: auto)
  --output-dir DIR           Output directory (default: outputs/benchmark/lmeval)

  -h, --help                 Show this help message

Authentication (for remote APIs):
  Set OPENAI_API_KEY environment variable before running:
    export OPENAI_API_KEY=sk-xxx

Prerequisites:
  1. Deploy vLLM server first:
     bash tools/serve/deploy_vllm.sh <model_path> -d 0 -t 1

  2. Wait for server ready, then run evaluation:
     bash tools/eval/run_lmeval.sh <model_path> --tasks wikitext

Examples:
  # Basic usage (server already running on port 8080)
  $0 outputs/qwen-int8 --tasks wikitext

  # Multiple tasks
  $0 outputs/model --tasks arc_challenge,hellaswag,gsm8k

  # Custom server URL
  $0 Qwen/Qwen2.5-0.5B --url http://192.168.1.100:8000/v1/completions

  # Remote API (e.g., DeepSeek) - set API key first
  export OPENAI_API_KEY=sk-xxx
  $0 deepseek-chat --url https://api.deepseek.com/v1/completions --model-name deepseek-chat

  # Specify model name explicitly
  $0 ./local-model --model-name Qwen/Qwen2.5-0.5B --tasks wikitext
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
MODEL_PATH=""
MODEL_NAME=""
URL=""
PORT=8080
TASKS="wikitext"
FEWSHOT=0
BATCH_SIZE="auto"
OUTPUT_DIR="outputs/benchmark/lmeval"

POSITIONAL_ARGS=()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --fewshot|--num-fewshot) FEWSHOT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# Handle positional MODEL_PATH
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
if [[ -z "$MODEL_PATH" ]]; then
    log_error "Model path is required"
    usage
    exit 1
fi

# Derive model name if not provided
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$MODEL_PATH"
fi

# Build URL if not provided
if [[ -z "$URL" ]]; then
    URL="http://127.0.0.1:${PORT}/v1/completions"
fi

# ------------------------------------------------------------------------------
# Build Output Path
# ------------------------------------------------------------------------------
MODEL_DIRNAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(get_timestamp)
SAVE_DIR="${OUTPUT_DIR}/${MODEL_DIRNAME}"
ensure_dir "$SAVE_DIR"
OUTPUT_FILE="${SAVE_DIR}/${TASKS//,/_}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Build Model Args for local-completions
# ------------------------------------------------------------------------------
# For local-completions: 'model' is used for tokenizer path AND API model name
# Use MODEL_PATH for tokenizer (local path), MODEL_NAME for API requests
MODEL_ARGS="model=${MODEL_PATH}"
MODEL_ARGS+=",base_url=${URL}"
MODEL_ARGS+=",tokenized_requests=False"
MODEL_ARGS+=",trust_remote_code=True"

# Log the distinction for debugging
log_debug "Tokenizer path: ${MODEL_PATH}"
log_debug "API model name: ${MODEL_NAME}"

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "LM-Evaluation-Harness (API Mode)"
log_info "Tokenizer" "$MODEL_PATH"
log_info "API Model" "$MODEL_NAME"
log_info "API URL" "$URL"
log_info "Tasks" "$TASKS"
log_info "Fewshot" "$FEWSHOT"
log_info "Output" "$OUTPUT_FILE"

# ------------------------------------------------------------------------------
# Verify Server Connectivity
# ------------------------------------------------------------------------------
log_info "Checking" "Server connectivity..."
# Extract base URL for health check (remove /v1/completions suffix)
HEALTH_URL=$(echo "$URL" | sed 's|/v1/.*$||')/health
if ! curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q "200"; then
    log_error "Server health check failed. Is vLLM server running?"
    log_tip "Deploy server first: bash tools/serve/deploy_vllm.sh <model> -d 0 -t 1"
    exit 1
fi

# ------------------------------------------------------------------------------
# Verify Dependencies
# ------------------------------------------------------------------------------
python -c "import lm_eval" 2>/dev/null || {
    log_error "'lm-evaluation-harness' not found. Install with: pip install lm-eval"
    exit 1
}

# ------------------------------------------------------------------------------
# Execute Evaluation
# ------------------------------------------------------------------------------
log_info "Launching" "lm_eval with local-completions backend..."

# Disable torch extension autoload to avoid torch_npu errors on systems with NPU
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

PYTHONUNBUFFERED=1 lm_eval \
    --model local-completions \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$FEWSHOT" \
    --output_path "$OUTPUT_FILE"

log_success "Evaluation completed"
log_info "Results" "$(dirname "$OUTPUT_FILE")"
