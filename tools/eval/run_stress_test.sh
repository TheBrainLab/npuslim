#!/bin/bash

# ==============================================================================
# vLLM Stress Test Script (External Server)
# ==============================================================================
# Usage: bash run_stress_test.sh [MODEL_PATH] [OPTIONS]
# ==============================================================================
# This script runs evalscope perf stress test against a running vLLM server.
# Deploy server first using: bash tools/serve/deploy_vllm.sh <model> -d 0 -t 1
# ==============================================================================

# Resolve paths independent of working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/common.sh"

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Run evalscope perf stress test against a running vLLM server.
Requires a running vLLM server (use tools/serve/deploy_vllm.sh first).

Arguments:
  MODEL_PATH               Path to model for tokenizer (required)

Options:
  --model-name NAME           Model name sent to API (default: derived from MODEL_PATH)
  --url URL                   Endpoint URL (default: http://127.0.0.1:8080/v1/chat/completions)
  --port PORT                 Server port for auto URL (default: 8080)

Benchmark Options:
  --parallel LIST             Concurrency levels (default: "1 10 50 100")
  --total-requests LIST       Requests per level (default: "10 50 100 200")
  --prompt-length INT         Prompt length (default: 1024)
  --max-tokens INT            Generation length (default: 1024)

Output Options:
  --output-dir DIR            Output directory (default: outputs/benchmark/stress_test)

  -h, --help                  Show this help message

Authentication (for remote APIs):
  Set OPENAI_API_KEY environment variable before running:
    export OPENAI_API_KEY=sk-xxx

Prerequisites:
  1. Deploy vLLM server first:
     bash tools/serve/deploy_vllm.sh <model_path> -d 0 -t 1

  2. Wait for server ready, then run stress test:
     bash tools/eval/run_stress_test.sh <model_path>

Examples:
  # Basic usage (server already running on port 8080)
  $0 outputs/qwen-int8

  # Custom parallelism levels
  $0 outputs/qwen-int8 --parallel "1 16 32" --total-requests "20 100 200"

  # Custom server URL
  $0 ./local-model --url http://192.168.1.100:8000/v1/chat/completions

  # Remote API (e.g., DeepSeek) - set API key first
  export OPENAI_API_KEY=sk-xxx
  $0 deepseek-chat --url https://api.deepseek.com/v1/chat/completions --model-name deepseek-chat
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
MODEL_PATH=""
MODEL_NAME=""
PORT=8080
URL=""

PARALLEL_LIST="1 10 50 100"
TOTAL_REQUESTS="10 50 100 200"
PROMPT_LENGTH=1024
MAX_TOKENS=1024

OUTPUT_DIR="outputs/benchmark/stress_test"

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
        --parallel) PARALLEL_LIST="$2"; shift 2 ;;
        --number|--total-requests) TOTAL_REQUESTS="$2"; shift 2 ;;
        --prompt-length|--prompt-len) PROMPT_LENGTH="$2"; shift 2 ;;
        --max-tokens|--gen-len) MAX_TOKENS="$2"; shift 2 ;;
        --output-dir|--outputs-dir) OUTPUT_DIR="$2"; shift 2 ;;
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

# Check dependencies
require_command "evalscope" "'evalscope' not found. Install with: pip install evalscope[perf]"

# Set default URL if not provided
[[ -z "$URL" ]] && URL="http://127.0.0.1:${PORT}/v1/chat/completions"

# ------------------------------------------------------------------------------
# Setup Output Directory
# ------------------------------------------------------------------------------
ensure_dir "$OUTPUT_DIR"
MODEL_DIRNAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(get_timestamp)
BENCHMARK_LOG="${OUTPUT_DIR}/${MODEL_DIRNAME}_${TIMESTAMP}.log"

# ------------------------------------------------------------------------------
# Server Health Check
# ------------------------------------------------------------------------------
log_header "Stress Test (External Server)"
log_info "Model" "$MODEL_NAME"
log_info "URL" "$URL"
log_info "Parallel" "$PARALLEL_LIST"
log_info "Requests" "$TOTAL_REQUESTS"
log_info "Prompt" "$PROMPT_LENGTH tokens"
log_info "Max Gen" "$MAX_TOKENS tokens"
log_info "Log" "$BENCHMARK_LOG"

log_info "Checking" "Server connectivity..."
HEALTH_URL="http://127.0.0.1:${PORT}/health"
http_code=$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 5 -m 10 "$HEALTH_URL" 2>/dev/null || echo "000")

if [[ "$http_code" != "200" ]]; then
    log_error "Server not responding (HTTP $http_code). Is vLLM server running on port $PORT?"
    log_tip "Deploy server first: bash tools/serve/deploy_vllm.sh <model> -d 0 -t 1"
    exit 1
fi
log_success "Server is UP (HTTP 200)"

# ------------------------------------------------------------------------------
# Run Benchmark
# ------------------------------------------------------------------------------
log_header "Running Stress Test..."
echo ""

evalscope perf \
    --parallel $PARALLEL_LIST \
    --number $TOTAL_REQUESTS \
    --model "$MODEL_NAME" \
    --tokenizer-path "$MODEL_PATH" \
    --url "$URL" \
    --api openai \
    --dataset random \
    --min-prompt-length "$PROMPT_LENGTH" \
    --max-prompt-length "$PROMPT_LENGTH" \
    --min-tokens "$MAX_TOKENS" \
    --max-tokens "$MAX_TOKENS" \
    --prefix-length 0 \
    --outputs-dir "$OUTPUT_DIR" \
    --extra-args '{"ignore_eos": true}' \
    2>&1 | tee "$BENCHMARK_LOG"

EXIT_CODE=${PIPESTATUS[0]}

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Benchmark completed successfully!"
    log_info "Results" "$OUTPUT_DIR"
else
    log_warn "Benchmark finished with errors (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
