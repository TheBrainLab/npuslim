#!/bin/bash

# --- 1. 加载工具函数 (如果有) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env_utils.sh" ]; then
    source "${SCRIPT_DIR}/env_utils.sh"
else
    # Fallback: 如果没有 env_utils.sh，内置检测逻辑
    detect_device() {
        if command -v npu-smi &> /dev/null || [ -c /dev/davinci0 ]; then echo "npu";
        elif command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then echo "gpu";
        else echo "cpu"; fi
    }
    setup_env() {
        local type=$1; local devs=$2
        if [[ "$type" == "npu" ]]; then export ASCEND_RT_VISIBLE_DEVICES=$devs; export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False;
        elif [[ "$type" == "gpu" ]]; then export CUDA_VISIBLE_DEVICES=$devs; fi
    }
fi

usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Options:
  --model-path PATH         Model path (Optional if provided as first arg)
  --device-type TYPE        'npu' or 'gpu' (Auto-detected if not set)
  --port PORT               Service port (default: 8080)
  -d, --devices DEVICES     Device IDs (default: 0,1)
  -t, --tensor-parallel     TP size (default: 2)
  -p, --pipeline-parallel   PP size (default: 1)
  -g, --memory-util         Memory utilization (default: 0.9)
  --max-model-len           Max model length (default: 4096)
  -q, --quantization [M]    Quantization method
  --media-path PATH         Allowed local media path for VLM
  --wait                    Wait until the server is healthy
  -h, --help                Show this message
EOF
}

# --- 2. 默认参数 ---
VISIBLE_DEVICES="0,1"
INFERENCE_TP_SIZE=2
PIPELINE_PARALLEL_SIZE=1
PORT=8080
MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=4096
QUANT_METHOD=""
MEDIA_PATH=""
WAIT_FOR_READY=false
POSITIONAL_ARGS=()

# --- 3. 参数解析 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --device-type) DEVICE_TYPE="$2"; shift 2 ;;
        -d|--devices) VISIBLE_DEVICES="$2"; shift 2 ;;
        -t|--tensor-parallel) INFERENCE_TP_SIZE="$2"; shift 2 ;;
        -p|--pipeline-parallel-size) PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;; # 补回了 PP 参数
        --port) PORT="$2"; shift 2 ;;
        -g|--gpu-memory|--memory-util) MEMORY_UTILIZATION="$2"; shift 2 ;; # 兼容旧参数名
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "$2" && "$2" != -* ]]; then QUANT_METHOD="$2"; shift 2
            else QUANT_METHOD="auto"; shift 1; fi ;;
        --media-path) MEDIA_PATH="$2"; shift 2 ;;
        --wait) WAIT_FOR_READY=true; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) POSITIONAL_ARGS+=("$1"); shift ;; # 关键：收集位置参数
    esac
done

# --- 4. 恢复位置参数逻辑 (关键修复) ---
# 如果 MODEL_PATH 为空，且有位置参数，则取第一个位置参数作为路径
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

# --- 5. 校验与环境准备 ---
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Error: Missing model path."
    usage
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# 自动检测设备
[[ -z "$DEVICE_TYPE" ]] && DEVICE_TYPE=$(detect_device)
setup_env "$DEVICE_TYPE" "$VISIBLE_DEVICES"

# 处理量化默认值
if [[ "$QUANT_METHOD" == "auto" ]]; then
    [[ "$DEVICE_TYPE" == "npu" ]] && QUANT_METHOD="ascend" || QUANT_METHOD=""
fi

# --- 6. 构建命令 ---
EXTRA_PARAMS=()
[ -n "$QUANT_METHOD" ] && EXTRA_PARAMS+=("--quantization" "$QUANT_METHOD")
[ -n "$MEDIA_PATH" ] && EXTRA_PARAMS+=("--allowed-local-media-path" "$MEDIA_PATH")
[ "$DEVICE_TYPE" == "npu" ] && EXTRA_PARAMS+=("--device" "npu")

echo "🔍 Using Model Path: $MODEL_PATH"
echo "🚀 Starting vLLM on ${DEVICE_TYPE^^} (Port: ${PORT})..."

# --- 7. 执行启动 ---
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --tensor-parallel-size "${INFERENCE_TP_SIZE}" \
    --trust-remote-code \
    --gpu-memory-utilization "${MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${EXTRA_PARAMS[@]}" &

VLLM_PID=$!

# --- 8. 健康检查 (可选) ---
if [ "$WAIT_FOR_READY" = true ]; then
    echo "⌛ Waiting for vLLM to become healthy..."
    while true; do
        if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
            echo "✅ vLLM is ready!"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ vLLM process died. Check logs above."
            exit 1
        fi
        sleep 5
    done
fi

wait $VLLM_PID