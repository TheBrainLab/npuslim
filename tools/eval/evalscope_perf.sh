#!/bin/bash

# --- Helper Functions ---
usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Description:
    Runs a stress test using 'evalscope perf' against a local vLLM server.
    Supports passing MODEL_PATH directly as the first argument.

Options:
  --model-path PATH         Path to the model/tokenizer (Optional if provided as first arg)
  --url URL                 Endpoint URL (default: http://127.0.0.1:8080/v1/chat/completions)
  --parallel LIST           Space-separated list of concurrency levels (default: "1 10 50 100")
  --prompt-len INT          Fixed input prompt length (default: 1024)
  --gen-len INT             Fixed generation length (default: 1024)
  --total-requests LIST     Total requests per concurrency level. (default: "10 50 100 200")
  --outputs-dir DIR         Directory to save logs and results (default: outputs/benchmark/stress_test)
  -h, --help                Show this help message

Example:
  bash $0 outputs/qwen-int8 --parallel "1 16 32" --outputs-dir my_results
EOF
}

# --- Default Config ---
MODEL_PATH=""
URL="http://127.0.0.1:8080/v1/chat/completions"
PARALLEL_LIST="1 10 50 100" 
NUMBER_LIST="10 50 100 200"
PROMPT_LEN=1024
GEN_LEN=1024
OUTPUTS_DIR="outputs/benchmark/stress_test"  # 默认输出目录

POSITIONAL_ARGS=()

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --url)
            URL="$2"; shift 2 ;;
        --parallel)
            PARALLEL_LIST="$2"; shift 2 ;;
        --number|--total-requests)
            NUMBER_LIST="$2"; shift 2 ;;
        --prompt-len)
            PROMPT_LEN="$2"; shift 2 ;;
        --gen-len)
            GEN_LEN="$2"; shift 2 ;;
        --outputs-dir)
            OUTPUTS_DIR="$2"; shift 2 ;;  # 新增参数解析
        -h|--help)
            usage; exit 0 ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# --- Handle Positional Arguments ---
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

# --- Validation ---
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Error: Missing model path."
    usage
    exit 1
fi

if ! command -v evalscope &> /dev/null; then
    echo "❌ Error: 'evalscope' command not found. Please install it:"
    echo "   pip install evalscope"
    exit 1
fi

# --- Execution ---
# 确保输出目录存在
mkdir -p "$OUTPUTS_DIR"

echo "🚀 Starting Stress Test with EvalScope..."
echo "   Model:       $MODEL_PATH"
echo "   URL:         $URL"
echo "   Parallel:    $PARALLEL_LIST"
echo "   Requests:    $NUMBER_LIST"
echo "   Output Dir:  $OUTPUTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
# 日志文件也放进 Output Dir
LOG_FILE="${OUTPUTS_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"

echo "📝 Logging output to: $LOG_FILE"

# 关键：增加了 --outputs-dir 参数传递给 evalscope
evalscope perf \
  --parallel $PARALLEL_LIST \
  --number $NUMBER_LIST \
  --model "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --url "$URL" \
  --api openai \
  --dataset random \
  --min-prompt-length $PROMPT_LEN \
  --max-prompt-length $PROMPT_LEN \
  --min-tokens $GEN_LEN \
  --max-tokens $GEN_LEN \
  --prefix-length 0 \
  --outputs-dir "$OUTPUTS_DIR" \
  --extra-args '{"ignore_eos": true}' \
  | tee "$LOG_FILE"

echo "✅ Stress test completed. Results saved to $OUTPUTS_DIR"