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
