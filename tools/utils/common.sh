#!/bin/bash

# ==============================================================================
# NPUSlim Common Bash Library
# ==============================================================================
# Usage: source "${SCRIPT_DIR}/../utils/common.sh"
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Path Utilities
# ------------------------------------------------------------------------------

# Get project root directory (works from any subdirectory)
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    dirname "$(dirname "$script_dir")"  # tools/utils/ -> tools/ -> project root
}

# ------------------------------------------------------------------------------
# Hardware Detection
# ------------------------------------------------------------------------------

# Detect hardware platform: npu, gpu, or cpu
detect_device() {
    if command -v npu-smi &> /dev/null || [ -c /dev/davinci0 ]; then
        echo "npu"
    elif command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------

# Setup environment variables for device type
# Usage: setup_env <device_type> <devices> [tp_size] [hccl_port]
# Example: setup_env "npu" "0,1" 2 60000
setup_env() {
    local device_type="${1:-npu}"
    local devices="${2:-0}"
    local tp_size="${3:-1}"
    local hccl_port="${4:-60000}"

    # Register NPUSlim plugin
    export NPUSLIM_PLUGIN_ENABLE=1

    if [[ "$device_type" == "npu" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="$devices"
        export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
        export HCCL_INTRA_PCIE_ENABLE=1
        export HCCL_INTRA_ROCE_ENABLE=0
        export HCCL_BUFFSIZE=512
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_BASE_PORT="$hccl_port"
        export TASK_QUEUE_ENABLE=1

        # Enable FlashComm1 only for multi-card (TP > 1)
        if [[ "$tp_size" -gt 1 ]]; then
            export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        fi

    elif [[ "$device_type" == "gpu" ]]; then
        export CUDA_VISIBLE_DEVICES="$devices"
    fi
}

# ------------------------------------------------------------------------------
# Logging Utilities
# ------------------------------------------------------------------------------

# Colors for terminal output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# Print section header
# Usage: log_header "Section Title"
log_header() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

# Print info message with optional key-value pair
# Usage: log_info "message"  OR  log_info "Key" "Value"
log_info() {
    if [[ $# -eq 2 ]]; then
        printf "   ${BOLD}%-12s${NC} %s\n" "$1:" "$2"
    else
        echo "   $1"
    fi
}

# Print success message
log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print error message and exit
log_error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

# Print warning message
log_warn() {
    echo -e "${YELLOW}⚠️  Warning: $1${NC}"
}

# Print debug message (only if DEBUG is set)
log_debug() {
    if [[ -n "${DEBUG:-}" ]]; then
        echo -e "${CYAN}🐛 [DEBUG] $1${NC}"
    fi
}

# Print tip/hint message
log_tip() {
    echo -e "${BLUE}💡 Tip: $1${NC}"
}

# ------------------------------------------------------------------------------
# Validation Utilities
# ------------------------------------------------------------------------------

# Check if a command exists
# Usage: require_command "python" "Python is required but not found"
require_command() {
    local cmd="$1"
    local msg="${2:-Command '$cmd' not found}"
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$msg"
    fi
}

# Check if a file exists
# Usage: require_file "/path/to/file" "File not found"
require_file() {
    local path="$1"
    local msg="${2:-File not found: $path}"
    if [[ ! -f "$path" ]]; then
        log_error "$msg"
    fi
}

# Check if a directory exists, create if not
# Usage: ensure_dir "/path/to/dir"
ensure_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        mkdir -p "$path"
        log_debug "Created directory: $path"
    fi
}

# ------------------------------------------------------------------------------
# Time Utilities
# ------------------------------------------------------------------------------

# Get current timestamp
get_timestamp() {
    date +%Y%m%d_%H%M%S
}

# Calculate elapsed time from start (requires START_TIME to be set)
elapsed_time() {
    local current=$(date +%s)
    local start="${START_TIME:-$(date +%s)}"
    echo $((current - start))
}

# ------------------------------------------------------------------------------
# vLLM Server Utilities
# ------------------------------------------------------------------------------

# Check vLLM server health and optionally verify model identity
# Usage: check_vllm_server [OPTIONS]
#   --port PORT           Server port (default: 8080)
#   --model MODEL_PATH    Expected model path/name (optional, for verification)
#   --strict              Fail if model mismatch (default: warn only)
#
# Returns:
#   0 on success, 1 on failure
#   Sets global variables: VLLM_MODEL_ID (served model ID if available)
#
# Example:
#   check_vllm_server --port 8080 --model "./outputs/qwen-int8"
#   check_vllm_server --port 8080  # Health check only
check_vllm_server() {
    local port=8080
    local expected_model=""
    local strict=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port) port="$2"; shift 2 ;;
            --model) expected_model="$2"; shift 2 ;;
            --strict) strict=true; shift ;;
            *) shift ;;
        esac
    done

    local base_url="http://127.0.0.1:${port}"

    # Step 1: Basic health check
    local health_url="${base_url}/health"
    local http_code
    http_code=$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 5 -m 10 "$health_url" 2>/dev/null || echo "000")

    if [[ "$http_code" != "200" ]]; then
        log_error "Server not responding (HTTP $http_code). Is vLLM server running on port $port?"
        log_tip "Deploy server first: bash tools/serve/deploy_vllm.sh <model> -d 0 -t 1"
        return 1
    fi

    log_success "Server is UP (HTTP 200)"

    # Step 2: Model verification (if requested)
    if [[ -n "$expected_model" ]]; then
        local models_url="${base_url}/v1/models"
        local models_json

        models_json=$(curl -s --connect-timeout 5 -m 10 "$models_url" 2>/dev/null)

        if [[ -z "$models_json" ]]; then
            log_warn "Could not fetch model info from ${models_url}"
            log_warn "Skipping model verification (server may not support /v1/models)"
            return 0
        fi

        # Extract model ID from response (handles both single and multiple models)
        # Response format: {"object":"list","data":[{"id":"model-name","object":"model",...}]}
        VLLM_MODEL_ID=$(echo "$models_json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0].get('id', ''))
    elif 'id' in data:
        print(data.get('id', ''))
except:
    pass
" 2>/dev/null || echo "")

        if [[ -z "$VLLM_MODEL_ID" ]]; then
            log_warn "Could not parse model ID from server response"
            log_debug "Response: ${models_json:0:200}..."
            return 0
        fi

        log_info "Served Model" "$VLLM_MODEL_ID"

        # Normalize paths for comparison (handle trailing slashes, relative paths)
        local norm_expected norm_served
        norm_expected=$(cd "$expected_model" 2>/dev/null && pwd || echo "$expected_model")
        norm_served="$VLLM_MODEL_ID"

        # Try to normalize served model path if it looks like a path
        if [[ "$norm_served" == /* ]] || [[ "$norm_served" == ./* ]]; then
            norm_served=$(cd "$norm_served" 2>/dev/null && pwd || echo "$norm_served")
        fi

        # Check for match (exact or basename match)
        local basename_expected basename_served
        basename_expected=$(basename "$norm_expected")
        basename_served=$(basename "$norm_served")

        if [[ "$norm_expected" == "$norm_served" ]] || [[ "$basename_expected" == "$basename_served" ]]; then
            log_success "Model verified: '$basename_expected' matches served model"
        else
            local msg="Model mismatch! Expected '$basename_expected' but server is running '$basename_served'"
            if [[ "$strict" == true ]]; then
                log_error "$msg"
                return 1
            else
                log_warn "$msg"
                log_warn "Evaluation will proceed with the SERVED model, not the expected one!"
            fi
        fi
    fi

    return 0
}
