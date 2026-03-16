#!/bin/bash

# ==============================================================================
# vLLM Server Deployment Script
# ==============================================================================
# Usage: bash deploy_vllm.sh [MODEL_PATH] [OPTIONS]
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

Deploy a vLLM inference server on NPU or GPU.

Arguments:
  MODEL_PATH               Path to model (required, can be positional)

Options:
  -d, --devices DEVICES       Device IDs (default: 0,1)
  -t, --tp SIZE               Tensor parallel size (default: 2)
  -p, --pp SIZE               Pipeline parallel size (default: 1)
  --port PORT                 Service port (default: 8080)
  --gpu-memory UTIL           GPU memory utilization (default: 0.8)
  --max-model-len LEN         Max model length (default: 4096)
  --quantization [METHOD]     Quantization method (auto-detect on NPU)
  --media-path PATH           Allowed local media path for VLM
  --wait                      Wait until server is healthy
  --enforce-eager             Use eager execution mode
  -h, --help                  Show this help message

Examples:
  $0 Qwen/Qwen3-0.6B -d 0 -t 1
  $0 outputs/qwen-int8 -d 0,1 -t 2 --wait
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
DEVICES="0,1"
TP_SIZE=2
PP_SIZE=1
PORT=8080
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
QUANT_METHOD=""
MEDIA_PATH=""
WAIT_FOR_READY=false
ENFORCE_EAGER=false
MODEL_PATH=""
DEVICE_TYPE=""

POSITIONAL_ARGS=()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --device-type) DEVICE_TYPE="$2"; shift 2 ;;
        -d|--devices) DEVICES="$2"; shift 2 ;;
        -t|--tp|--tensor-parallel) TP_SIZE="$2"; shift 2 ;;
        -p|--pp|--pipeline-parallel) PP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -g|--gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="auto"; shift 1
            fi ;;
        --media-path) MEDIA_PATH="$2"; shift 2 ;;
        --wait) WAIT_FOR_READY=true; shift 1 ;;
        --enforce-eager) ENFORCE_EAGER=true; shift 1 ;;
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

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
[[ -z "$DEVICE_TYPE" ]] && DEVICE_TYPE=$(detect_device)
setup_env "$DEVICE_TYPE" "$DEVICES"

# Auto-detect quantization method
if [[ "$QUANT_METHOD" == "auto" ]]; then
    if [[ "$DEVICE_TYPE" == "npu" ]]; then
        QUANT_METHOD="ascend"
    else
        QUANT_METHOD=""
    fi
fi

# ------------------------------------------------------------------------------
# Build Command
# ------------------------------------------------------------------------------
EXTRA_PARAMS=()
[[ -n "$QUANT_METHOD" ]] && EXTRA_PARAMS+=("--quantization" "$QUANT_METHOD")
[[ -n "$MEDIA_PATH" ]] && EXTRA_PARAMS+=("--allowed-local-media-path" "$MEDIA_PATH")
[[ "$ENFORCE_EAGER" == true ]] && EXTRA_PARAMS+=("--enforce-eager")

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "vLLM Server Deployment"
log_info "Model" "$MODEL_PATH"
log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
log_info "Port" "$PORT"
log_info "TP/PP" "$TP_SIZE / $PP_SIZE"
log_info "Memory" "$MEM_UTIL"
if [[ -n "$QUANT_METHOD" ]]; then
    log_info "Quant" "$QUANT_METHOD"
fi

# ------------------------------------------------------------------------------
# Launch Server
# ------------------------------------------------------------------------------
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    "${EXTRA_PARAMS[@]}" &

SERVER_PID=$!

# ------------------------------------------------------------------------------
# Health Check (Optional)
# ------------------------------------------------------------------------------
if [[ "$WAIT_FOR_READY" == true ]]; then
    log_info "Waiting for server to become healthy..."
    START_TIME=$(date +%s)

    while true; do
        if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
            log_success "vLLM server is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log_error "vLLM process died. Check logs above."
        fi
        sleep 5
    done
fi

wait $SERVER_PID
