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

Run lm-evaluation-harness benchmarks on NPU or GPU.

Arguments:
  MODEL_PATH               Path to model (required, can be positional)

Options:
  --backend TYPE              Backend: 'vllm' or 'hf' (default: vllm)
  --tasks LIST                Comma-separated tasks (default: wikitext)
  --fewshot N                 Number of few-shot examples (default: 0)
  --batch-size SIZE           Batch size or 'auto' (default: auto)
  --output-dir DIR            Output directory (default: outputs/lmeval)

  Hardware Options:
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)
  --gpu-memory UTIL           GPU memory utilization (default: 0.8)
  --max-model-len LEN         Max model length (default: 4096)
  -q, --quantization [TYPE]   Quantization method (auto-set on NPU)

  -h, --help                  Show this help message

Examples:
  $0 outputs/qwen-int8 --tasks wikitext --fewshot 5 -d 0,1 -t 2
  $0 outputs/model --tasks arc_easy,hellaswag -q
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
MODEL_PATH=""
BACKEND="vllm"
TASKS="wikitext"
FEWSHOT=0
BATCH_SIZE="auto"
OUTPUT_DIR="outputs/lmeval"

DEVICES="0"
TP_SIZE=1
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
QUANT_METHOD=""
DEVICE_TYPE=""

POSITIONAL_ARGS=()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --fewshot|--num-fewshot) FEWSHOT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -d|--devices) DEVICES="$2"; shift 2 ;;
        -t|--tp) TP_SIZE="$2"; shift 2 ;;
        --gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="ascend"; shift 1
            fi ;;
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

# ------------------------------------------------------------------------------
# Build Output Path
# ------------------------------------------------------------------------------
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(get_timestamp)
SAVE_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
ensure_dir "$SAVE_DIR"
OUTPUT_FILE="${SAVE_DIR}/${TASKS//,/_}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Build Model Args
# ------------------------------------------------------------------------------
MODEL_ARGS="pretrained=${MODEL_PATH},trust_remote_code=True"

if [[ "$BACKEND" == "vllm" ]]; then
    MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
    MODEL_ARGS+=",gpu_memory_utilization=${MEM_UTIL}"
    MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN}"
    MODEL_ARGS+=",dtype=auto"
    if [[ -n "$QUANT_METHOD" ]]; then
        MODEL_ARGS+=",quantization=${QUANT_METHOD}"
    fi
elif [[ "$BACKEND" == "hf" ]]; then
    if [[ "$TP_SIZE" -gt 1 ]]; then
        MODEL_ARGS+=",parallelize=True"
    fi
fi

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "LM-Evaluation-Harness"
log_info "Model" "$MODEL_PATH"
log_info "Backend" "$BACKEND"
log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
log_info "Tasks" "$TASKS"
log_info "Fewshot" "$FEWSHOT"
log_info "Output" "$OUTPUT_FILE"

# ------------------------------------------------------------------------------
# Verify Dependencies
# ------------------------------------------------------------------------------
python -c "import npuslim; import lm_eval" 2>/dev/null || {
    log_error "'npuslim' or 'lm-evaluation-harness' not found in Python environment"
}

# ------------------------------------------------------------------------------
# Execute Evaluation
# ------------------------------------------------------------------------------
log_info "Launching lm_eval with NPUSlim plugin..."

PYTHONUNBUFFERED=1 lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$FEWSHOT" \
    --output_path "$OUTPUT_FILE"

log_success "Evaluation completed"
log_info "Results" "$(dirname "$OUTPUT_FILE")"
