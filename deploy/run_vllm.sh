#!/bin/bash

usage() {
    cat << EOF
Usage: $0 --model-path <path> [OPTIONS]

Options:
  --model-path                 Model path (Required)
  --port PORT                  Service port (default: 8080)
  -d, --devices DEVICES        Ascend devices to use (default: 0,1)
  -t, --tensor-parallel SIZE   Tensor parallel size (default: 2)
  -p, --pipeline-parallel-size Pipeline parallel size (default: 1)
  -g, --gpu-memory UTILIZATION GPU memory utilization (default: 0.9)
  --max-model-len              Max model len (default: 32000)
  -q, --quantization [METHOD]  Enable quantization (default: ascend if flag used)
  --media-path PATH            Allowed local media path (Optional)
  -h, --help                   Show this help message

Examples:
  bash $0 --model-path /path/to/model -d 4,5 -t 2
EOF
}

# 基础默认值
VISIBLE_DEVICES="0,1"
INFERENCE_TP_SIZE=2
PIPELINE_PARALLEL_SIZE=1
PORT=8080
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096

# 可选参数初始化为空
QUANT_METHOD=""
MEDIA_PATH=""

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        -d|--devices)
            VISIBLE_DEVICES="$2"; shift 2 ;;
        -t|--tensor-parallel)
            INFERENCE_TP_SIZE="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        -g|--gpu-memory)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --pipeline-parallel-size)
            PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            # 如果后面跟着一个不以 - 开头的参数，则视其为量化方法名
            if [[ -n "$2" && "$2" != -* ]]; then
                QUANT_METHOD="$2"
                shift 2
            else
                # 否则使用默认的 ascend
                QUANT_METHOD="ascend"
                shift 1
            fi
            ;;
        --media-path)
            MEDIA_PATH="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"
if [[ -z "$MODEL_PATH" && -n "$1" ]]; then
    MODEL_PATH="$1"
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Missing required argument: --model-path"
    echo "Usage example: bash $0 --model-path /your/model/path"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# 构造 Python 启动的可选参数数组
EXTRA_PARAMS=()

# 处理量化参数
if [ -n "$QUANT_METHOD" ]; then
    EXTRA_PARAMS+=("--quantization" "$QUANT_METHOD")
fi

# 处理媒体路径
if [ -n "$MEDIA_PATH" ]; then
    EXTRA_PARAMS+=("--allowed-local-media-path" "$MEDIA_PATH")
fi

# 环境配置
# gpu
export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES
# npu
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# 执行启动
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --pipeline_parallel_size ${PIPELINE_PARALLEL_SIZE} \
    --tensor-parallel-size ${INFERENCE_TP_SIZE} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    "${EXTRA_PARAMS[@]}"