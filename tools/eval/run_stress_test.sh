#!/bin/bash

# ==============================================================================
# vLLM Stress Test Pipeline Script
# ==============================================================================
# Usage: bash run_stress_test.sh [MODEL_PATH] [OPTIONS]
# ==============================================================================
# This script:
#   1. Deploys a vLLM server
#   2. Waits for service readiness
#   3. Runs evalscope perf stress test
#   4. Cleans up processes automatically
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

Run a stress test pipeline: deploy vLLM server → run benchmark → cleanup.

Arguments:
  MODEL_PATH               Path to model (required, can be positional)

Server Options:
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)
  --port PORT                 Service port (default: 8080)
  --hccl-port HCCL_IF_BASE_PORT
                              HCCL base port for NPU communication (default: 60000)
  --gpu-memory UTIL           GPU memory utilization (default: 0.8)
  --max-model-len LEN         Max model length (default: 4096)
  -ep, --enable-expert-parallel
                              Enable expert parallelism for MoE models
  --compilation-config CONFIG
                              Compilation config (e.g., '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
  --enforce-eager             Use eager execution mode (disable graph capture)

Benchmark Options:
  --parallel LIST             Concurrency levels (default: "1 10 50 100")
  --total-requests LIST       Requests per level (default: "10 50 100 200")
  --prompt-length INT         Prompt length (default: 1024)
  --max-tokens INT            Generation length (default: 1024)
  --url URL                   Override endpoint URL (default: auto from --port)

Output Options:
  --output-dir DIR            Output directory (default: outputs/benchmark/stress_test)

  -h, --help                  Show this help message

Examples:
  $0 outputs/qwen-int8 -d 0,1 -t 2
  $0 outputs/qwen-int8 --parallel "1 16 32" --total-requests "20 100 200"
  $0 outputs/moe-model -d 0,1 -t 2 -ep --hccl-port 65000
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
MODEL_PATH=""
DEVICES="0"
TP_SIZE=1
PORT=8080
HCCL_PORT=""
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
COMPILATION_CONFIG=""
ENABLE_EP=false
ENFORCE_EAGER=false

PARALLEL_LIST="1 10 50 100"
TOTAL_REQUESTS="10 50 100 200"
PROMPT_LENGTH=1024
MAX_TOKENS=1024
URL=""

OUTPUT_DIR="outputs/benchmark/stress_test"

POSITIONAL_ARGS=()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        -d|--devices) DEVICES="$2"; shift 2 ;;
        -t|--tp|--tensor-parallel) TP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --hccl-port) HCCL_PORT="$2"; shift 2 ;;
        --gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -ep|--enable-expert-parallel) ENABLE_EP=true; shift 1 ;;
        --compilation-config) COMPILATION_CONFIG="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=true; shift 1 ;;
        --parallel) PARALLEL_LIST="$2"; shift 2 ;;
        --number|--total-requests) TOTAL_REQUESTS="$2"; shift 2 ;;
        --prompt-length|--prompt-len) PROMPT_LENGTH="$2"; shift 2 ;;
        --max-tokens|--gen-len) MAX_TOKENS="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
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

# Check dependencies
require_command "evalscope" "'evalscope' not found. Install with: pip install evalscope"

# Set default URL if not provided
[[ -z "$URL" ]] && URL="http://localhost:${PORT}/v1/chat/completions"

# ------------------------------------------------------------------------------
# Setup Output Directory
# ------------------------------------------------------------------------------
ensure_dir "$OUTPUT_DIR"
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(get_timestamp)
BENCHMARK_LOG="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"

# ------------------------------------------------------------------------------
# Cleanup Handler
# ------------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
    echo ""
    if [[ "$1" == "interrupt" ]]; then
        log_warn "Interrupted! Cleaning up..."
    else
        log_info "Cleaning up server process..."
    fi

    if [[ -n "$SERVER_PID" ]]; then
        # Find child processes (vLLM Python process)
        child_pids=$(pgrep -P "$SERVER_PID" 2>/dev/null || true)

        log_info "Killing" "Server wrapper (PID: $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true

        if [[ -n "$child_pids" ]]; then
            log_info "Killing" "vLLM processes (PIDs: $child_pids)"
            kill $child_pids 2>/dev/null || true
            sleep 2
            kill -9 $child_pids 2>/dev/null || true
        fi

        # Ensure port is free
        log_info "Freeing" "Port $PORT"
        lsof -t -i:"$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi

    log_success "Cleanup complete"
}

# Handle interrupts (Ctrl+C)
handle_interrupt() {
    cleanup "interrupt"
    exit 1
}
trap handle_interrupt INT TERM

# ------------------------------------------------------------------------------
# Stage 1: Deploy Server
# ------------------------------------------------------------------------------
log_header "Stage 1: Deploying vLLM Server"
log_info "Model" "$MODEL_PATH"
log_info "Port" "$PORT"
log_info "Devices" "$DEVICES"
log_info "TP" "$TP_SIZE"
if [[ -n "$HCCL_PORT" ]]; then
    log_info "HCCL Port" "$HCCL_PORT"
fi
if [[ "$ENABLE_EP" == true ]]; then
    log_info "Expert Parallel" "enabled"
fi
log_tip "Server logs saved to: $OUTPUT_DIR"

DEPLOY_SCRIPT="${PROJECT_ROOT}/tools/serve/deploy_vllm.sh"

if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    log_error "Deploy script not found: $DEPLOY_SCRIPT"
fi

# Build deploy command with optional parameters
DEPLOY_CMD=(
    bash "$DEPLOY_SCRIPT"
    "$MODEL_PATH"
    --port "$PORT"
    --devices "$DEVICES"
    --tp "$TP_SIZE"
    --gpu-memory "$MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --log-dir "$OUTPUT_DIR"
)

# Add optional parameters
[[ -n "$HCCL_PORT" ]] && DEPLOY_CMD+=(--hccl-port "$HCCL_PORT")
[[ "$ENABLE_EP" == true ]] && DEPLOY_CMD+=(--enable-expert-parallel)
[[ -n "$COMPILATION_CONFIG" ]] && DEPLOY_CMD+=(--compilation-config "$COMPILATION_CONFIG")
[[ "$ENFORCE_EAGER" == true ]] && DEPLOY_CMD+=(--enforce-eager)

"${DEPLOY_CMD[@]}" &

SERVER_PID=$!
log_info "Launched" "Server PID: $SERVER_PID"

# ------------------------------------------------------------------------------
# Stage 2: Health Check
# ------------------------------------------------------------------------------
log_info "Waiting" "Server health check..."
START_TIME=$(date +%s)

while true; do
    # Check if process is alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        # Find the latest server log file
        LATEST_LOG=$(ls -t "${OUTPUT_DIR}/${MODEL_NAME}"*.log 2>/dev/null | head -1)
        log_error "Server process died! Check logs in: $OUTPUT_DIR"
        if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
            echo "----------------- Log Snippet ($LATEST_LOG) -----------------"
            tail -n 20 "$LATEST_LOG"
        fi
        exit 1
    fi

    # Check HTTP health (with timeout to prevent hanging)
    http_code=$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 5 -m 10 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    http_code="${http_code// /}"  # Trim any whitespace

    if [[ "$http_code" == "200" ]]; then
        echo ""  # Clear the loading line
        log_success "Server is UP (HTTP 200)"
        break
    fi

    elapsed=$(elapsed_time)
    echo -ne "   ⏳ Loading... (${elapsed}s) | This may take a few minutes. [Ctrl+C] to abort \033[K\r"
    sleep 3
done

# ------------------------------------------------------------------------------
# Stage 3: Run Benchmark
# ------------------------------------------------------------------------------
log_header "Stage 2: Running Stress Test"
log_info "Parallel" "$PARALLEL_LIST"
log_info "Requests" "$TOTAL_REQUESTS"
log_info "Prompt" "$PROMPT_LENGTH tokens"
log_info "Max Gen" "$MAX_TOKENS tokens"
log_info "URL" "$URL"
log_info "Log" "$BENCHMARK_LOG"

evalscope perf \
    --parallel $PARALLEL_LIST \
    --number $TOTAL_REQUESTS \
    --model "$MODEL_PATH" \
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

# Cleanup after successful completion
cleanup "normal"
exit $EXIT_CODE
