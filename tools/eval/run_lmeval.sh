#!/bin/bash

# ==============================================================================
# LM-Evaluation-Harness Wrapper Script
# ==============================================================================
# Usage: bash run_lmeval.sh [MODEL_PATH] [OPTIONS]
# ==============================================================================
# Backends:
#   - vllm: Direct vLLM loading (fastest, requires GPU/NPU)
#   - hf:   HuggingFace backend (for comparison)
#   - api:  OpenAI-compatible API (flexible, supports remote servers)
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

Run lm-evaluation-harness benchmarks with multiple backend options.

Arguments:
  MODEL_PATH               Path or name of model (required, can be positional)

Backend Options:
  --backend TYPE              Backend type: 'vllm', 'hf', or 'api' (default: vllm)
                              - vllm: Direct vLLM loading (fastest)
                              - hf:   HuggingFace backend
                              - api:  OpenAI-compatible API (requires running server)

Evaluation Options:
  --tasks LIST                Comma-separated tasks (default: wikitext)
  --fewshot N                 Number of few-shot examples (default: 0)
  --batch-size SIZE           Batch size or 'auto' (default: auto)
  --output-dir DIR            Output directory (default: outputs/benchmark/lmeval)
  --limit N                   Limit number of samples per task (default: all)
  --log-samples               Save model outputs for debugging

Model & Hardware Options (vllm/api backends):
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)
  --max-model-len LEN         Max model length (default: 4096)

vLLM Backend Options:
  --hccl-port PORT            HCCL base port for NPU (default: 60000)
  --gpu-memory UTIL           GPU memory utilization (default: 0.8)
  -q, --quantization [TYPE]   Quantization method (auto-set on NPU)
  -ep, --enable-expert-parallel
                              Enable expert parallelism for MoE models
  --compilation-config CONFIG
                              Compilation config (e.g., '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
  --enforce-eager             Use eager execution mode (disable graph capture)

HuggingFace Backend Options:
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)

API Backend Options:
  --url URL                   API endpoint URL (default: http://127.0.0.1:\$PORT/v1/completions)
  --port PORT                 Server port (default: 8080)
  --model-name NAME           Model name sent to API (default: MODEL_PATH)
  --chat                      Use /v1/chat/completions endpoint with local-chat-completions model
                              Required for generative tasks (e.g. mmlu_generative) on chat-tuned models
  --apply-chat-template       Apply model's chat template via HuggingFace tokenizer
                              Works with all backends. Note: incompatible with loglikelihood-based
                              tasks (e.g. mmlu) -- use --chat + mmlu_generative instead

Authentication (for remote APIs):
  Set OPENAI_API_KEY environment variable before running:
    export OPENAI_API_KEY=sk-xxx

  -h, --help                  Show this help message

Examples:
  # vLLM backend (fastest, direct loading)
  $0 outputs/qwen-int8 --backend vllm --tasks wikitext -d 0
  $0 outputs/model --backend vllm --tasks arc_challenge,arc_easy,boolq,headqa_en,hellaswag,openbookqa,piqa,winogrande -d 0,1 -t 2

  # HuggingFace backend
  $0 outputs/qwen-int8 --backend hf --tasks wikitext -d 0

  # API backend (requires running server)
  bash tools/serve/deploy_vllm.sh outputs/qwen-int8 -d 0 -t 1
  $0 outputs/qwen-int8 --backend api --tasks wikitext

  # Remote API (e.g., DeepSeek)
  export OPENAI_API_KEY=sk-xxx
  $0 deepseek-chat --backend api --url https://api.deepseek.com/v1/completions
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
OUTPUT_DIR="outputs/benchmark/lmeval"
LIMIT=""
LOG_SAMPLES=false
OFFLINE=false

# Hardware options (vllm/hf backends)
DEVICES="0"
TP_SIZE=1
HCCL_PORT=""
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
QUANT_METHOD=""
COMPILATION_CONFIG=""
ENABLE_EP=false
ENFORCE_EAGER=false
DEVICE_TYPE=""

# API options (api backend)
API_URL=""
API_PORT=8080
MODEL_NAME=""
API_CHAT=false
APPLY_CHAT_TEMPLATE=false

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
        --limit) LIMIT="$2"; shift 2 ;;
        --log-samples) LOG_SAMPLES=true; shift 1 ;;
        # Hardware options
        -d|--devices) DEVICES="$2"; shift 2 ;;
        -t|--tp) TP_SIZE="$2"; shift 2 ;;
        --hccl-port) HCCL_PORT="$2"; shift 2 ;;
        --gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="ascend"; shift 1
            fi ;;
        -ep|--enable-expert-parallel) ENABLE_EP=true; shift 1 ;;
        --compilation-config) COMPILATION_CONFIG="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=true; shift 1 ;;
        # API options
        --url) API_URL="$2"; shift 2 ;;
        --port) API_PORT="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --chat) API_CHAT=true; shift 1 ;;
        --apply-chat-template) APPLY_CHAT_TEMPLATE=true; shift 1 ;;
        --offline) OFFLINE=true; shift 1 ;;
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

# Validate backend
case "$BACKEND" in
    vllm|hf|api) ;;
    *) log_error "Invalid backend '$BACKEND'. Must be: vllm, hf, or api" ;;
esac

# Derive model name if not provided (for api backend)
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$MODEL_PATH"
fi

# ------------------------------------------------------------------------------
# Environment Setup (vllm/hf backends only)
# ------------------------------------------------------------------------------
if [[ "$BACKEND" != "api" ]]; then
    [[ -z "$DEVICE_TYPE" ]] && DEVICE_TYPE=$(detect_device)
    HCCL_PORT="${HCCL_PORT:-60000}"
    setup_env "$DEVICE_TYPE" "$DEVICES" "$TP_SIZE" "$HCCL_PORT"

    # Avoid OpenMP thread pool conflicts in multi-process vLLM environment
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
fi

# vLLM defaults to fork workers, which is unsafe once the parent process
# has already touched CUDA. Force spawn unless the user explicitly overrides it.
if [[ "$BACKEND" == "vllm" && -z "${VLLM_WORKER_MULTIPROC_METHOD:-}" ]]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

# ------------------------------------------------------------------------------
# Build Output Path
# ------------------------------------------------------------------------------
TIMESTAMP=$(get_timestamp)
ensure_dir "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/${TASKS//,/_}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Build Model Args (Unified key=value format)
# ------------------------------------------------------------------------------
# Common args for all backends
MODEL_ARGS="pretrained=${MODEL_PATH}"

if [[ "$BACKEND" == "api" ]]; then
    # ---------------------------
    # API Backend
    # ---------------------------
    # Build URL if not provided
    if [[ -z "$API_URL" ]]; then
        if [[ "$API_CHAT" == true ]]; then
            API_URL="http://127.0.0.1:${API_PORT}/v1/chat/completions"
        else
            API_URL="http://127.0.0.1:${API_PORT}/v1/completions"
        fi
    fi

    MODEL_ARGS+=",base_url=${API_URL}"
    # When --model-name differs from --model-path, separate API model ID from tokenizer:
    #   pretrained = API model name (sent in requests)
    #   tokenizer  = local path (for tokenization)
    if [[ -n "$MODEL_NAME" && "$MODEL_NAME" != "$MODEL_PATH" ]]; then
        MODEL_ARGS="pretrained=${MODEL_NAME}"
        MODEL_ARGS+=",tokenizer=${MODEL_PATH}"
        MODEL_ARGS+=",base_url=${API_URL}"
    fi
    # tokenized_requests must be True when using --apply-chat-template,
    # otherwise apply_chat_template returns JsonChatStr which breaks _encode_pair
    if [[ "$APPLY_CHAT_TEMPLATE" == true ]]; then
        MODEL_ARGS+=",tokenized_requests=True"
    else
        MODEL_ARGS+=",tokenized_requests=False"
    fi
    MODEL_ARGS+=",max_length=${MAX_MODEL_LEN}"
    MODEL_ARGS+=",trust_remote_code=True"
    if [[ "$API_CHAT" == true ]]; then
        LM_EVAL_MODEL="local-chat-completions"
    else
        LM_EVAL_MODEL="local-completions"
    fi

elif [[ "$BACKEND" == "vllm" ]]; then
    # ---------------------------
    # vLLM Backend
    # ---------------------------
    MODEL_ARGS+=",trust_remote_code=True"
    MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
    MODEL_ARGS+=",gpu_memory_utilization=${MEM_UTIL}"
    MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN}"
    MODEL_ARGS+=",dtype=auto"

    if [[ -n "$QUANT_METHOD" ]]; then
        MODEL_ARGS+=",quantization=${QUANT_METHOD}"
    fi
    if [[ "$ENABLE_EP" == true ]]; then
        MODEL_ARGS+=",enable_expert_parallel=True"
    fi
    if [[ -n "$COMPILATION_CONFIG" ]]; then
        MODEL_ARGS+=",compilation_config=${COMPILATION_CONFIG}"
    fi
    if [[ "$ENFORCE_EAGER" == true ]]; then
        MODEL_ARGS+=",enforce_eager=True"
    fi
    LM_EVAL_MODEL="vllm"

elif [[ "$BACKEND" == "hf" ]]; then
    # ---------------------------
    # HuggingFace Backend
    # ---------------------------
    MODEL_ARGS+=",trust_remote_code=True"

    if [[ "$TP_SIZE" -gt 1 ]]; then
        MODEL_ARGS+=",parallelize=True"
    fi
    LM_EVAL_MODEL="hf"
fi

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "LM-Evaluation-Harness"
log_info "Model" "$MODEL_PATH"
log_info "Backend" "$BACKEND"

if [[ "$BACKEND" == "vllm" ]]; then
    log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
    log_info "TP Size" "$TP_SIZE"
    log_info "MP Method" "${VLLM_WORKER_MULTIPROC_METHOD:-default}"
    [[ -n "$QUANT_METHOD" ]] && log_info "Quant" "$QUANT_METHOD"
elif [[ "$BACKEND" == "hf" ]]; then
    log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
elif [[ "$BACKEND" == "api" ]]; then
    log_info "API URL" "$API_URL"
fi
log_info "Tasks" "$TASKS"
log_info "Fewshot" "$FEWSHOT"
log_info "Output" "$OUTPUT_FILE"
[[ -n "$LIMIT" ]] && log_info "Limit" "$LIMIT samples per task"
[[ "$LOG_SAMPLES" == true ]] && log_info "Log Samples" "enabled"
[[ "$APPLY_CHAT_TEMPLATE" == true ]] && log_info "Chat Template" "enabled"

# ------------------------------------------------------------------------------
# Verify Dependencies
# ------------------------------------------------------------------------------
require_command "lm_eval" "'lm-evaluation-harness' not found. Install with: pip install lm-eval[api]"

# ------------------------------------------------------------------------------
# Server Health Check (api backend only)
# ------------------------------------------------------------------------------
if [[ "$BACKEND" == "api" ]]; then
    log_info "Checking" "Server connectivity and model identity..."
    check_vllm_server --port "$API_PORT" --model "$MODEL_PATH"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# Execute Evaluation
# ------------------------------------------------------------------------------
log_header "Launching lm_eval..."
echo ""

# Disable torch extension autoload in API mode to avoid torch_npu errors
# In vllm/hf modes, we need the plugin system to work
if [[ "$BACKEND" == "api" ]]; then
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
fi

# Offline mode: use cached datasets/models only, no network access
if [[ "$OFFLINE" == true ]]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Build optional arguments (avoid set -e issues with conditional command substitution)
OPTIONAL_ARGS=()
[[ -n "$LIMIT" ]] && OPTIONAL_ARGS+=(--limit "$LIMIT")
[[ "$LOG_SAMPLES" == true ]] && OPTIONAL_ARGS+=(--log_samples)
[[ "$APPLY_CHAT_TEMPLATE" == true ]] && OPTIONAL_ARGS+=(--apply_chat_template)

PYTHONUNBUFFERED=1 lm_eval \
    --model "$LM_EVAL_MODEL" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$FEWSHOT" \
    --output_path "$OUTPUT_FILE" \
    "${OPTIONAL_ARGS[@]}"

log_success "Evaluation completed"
log_info "Results" "$(dirname "$OUTPUT_FILE")"
