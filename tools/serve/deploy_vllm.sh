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
  -d, --devices DEVICES         Device IDs (default: 0,1)
  -t, --tp SIZE                 Tensor parallel size (default: 2)
  -p, --pp SIZE                 Pipeline parallel size (default: 1)
  --port PORT                   Service port (default: 8080)
  --hccl-port HCCL_IF_BASE_PORT
                                HCCL base port for NPU communication (default: 60000)
  --gpu-memory UTIL             GPU memory utilization (default: 0.8)
  --max-model-len LEN           Max model length (default: 4096)
  --quantization [METHOD]       Quantization method (auto-detect on NPU)
  --compilation-config CONFIG   Compilation config (e.g., '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
  --media-path PATH             Allowed local media path for VLM
  -ep, --enable-expert-parallel
                                Enable expert parallelism for MoE models
  --enforce-eager               Use eager execution mode
  --log-dir DIR                 Directory to save logs (default: logs/vllm_serve)
  --no-log                      Disable logging to file
  --wait                        Wait until server is healthy
  -h, --help                    Show this help message

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
HCCL_PORT=""
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
QUANT_METHOD=""
COMPILATION_CONFIG=""
MEDIA_PATH=""
WAIT_FOR_READY=false
ENFORCE_EAGER=false
ENABLE_EP=false
MODEL_PATH=""
DEVICE_TYPE=""
LOG_DIR="logs/vllm_server"
NO_LOG=false

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
        --hccl-port) HCCL_PORT="$2"; shift 2 ;;
        -g|--gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="auto"; shift 1
            fi ;;
        --compilation-config) COMPILATION_CONFIG="$2"; shift 2 ;;
        --media-path) MEDIA_PATH="$2"; shift 2 ;;
        -ep|--enable-expert-parallel) ENABLE_EP=true; shift 1 ;;
        --wait) WAIT_FOR_READY=true; shift 1 ;;
        --enforce-eager) ENFORCE_EAGER=true; shift 1 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --no-log) NO_LOG=true; shift 1 ;;
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
# Logging Setup
# ------------------------------------------------------------------------------
if [[ "$NO_LOG" != true ]]; then
    LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"
    ensure_dir "$LOG_DIR"

    TIMESTAMP=$(get_timestamp)
    MODEL_NAME=$(basename "$MODEL_PATH")
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"

    # Redirect stdout and stderr to tee (both file and console)
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
[[ -z "$DEVICE_TYPE" ]] && DEVICE_TYPE=$(detect_device)
HCCL_PORT="${HCCL_PORT:-60000}"
setup_env "$DEVICE_TYPE" "$DEVICES" "$TP_SIZE" "$HCCL_PORT"

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
[[ -n "$COMPILATION_CONFIG" ]] && EXTRA_PARAMS+=("--compilation-config" "$COMPILATION_CONFIG")
[[ "$ENABLE_EP" == true ]] && EXTRA_PARAMS+=("--enable-expert-parallel")
[[ "$ENFORCE_EAGER" == true ]] && EXTRA_PARAMS+=("--enforce-eager")

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "vLLM Server Deployment"
log_info "Model" "$MODEL_PATH"
log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
log_info "Port" "$PORT"
if [[ -n "$HCCL_PORT" && "$DEVICE_TYPE" == "npu" ]]; then
    log_info "HCCL Port" "$HCCL_PORT"
fi
log_info "TP/PP" "$TP_SIZE / $PP_SIZE"
log_info "Memory" "$MEM_UTIL"
if [[ -n "$QUANT_METHOD" ]]; then
    log_info "Quant" "$QUANT_METHOD"
fi
if [[ "$ENABLE_EP" == true ]]; then
    log_info "Expert Parallel" "enabled"
fi
if [[ "$NO_LOG" != true ]]; then
    log_info "Log File" "$LOG_FILE"
fi

# Tips
echo ""
if [[ "$TP_SIZE" -gt 1 && "$ENABLE_EP" != true ]]; then
    log_tip "For MoE models, enable expert parallelism with -ep flag"
fi
if [[ "$DEVICE_TYPE" == "npu" ]]; then
    log_tip "If HCCL port binding fails, try --hccl-port with a different value"
    if [[ "$ENFORCE_EAGER" != true ]]; then
        log_tip "If graph capture fails, try --enforce-eager or --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"
    fi
fi

log_header "Starting vLLM Server"
echo ""

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
