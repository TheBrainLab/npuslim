#!/bin/bash

# --- 1. 加载工具函数 (复用之前的 env_utils) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # 回退两级到根目录

# 尝试加载 env_utils，如果没有则使用内置简易版
ENV_UTILS="$PROJECT_ROOT/tools/deploy/env_utils.sh"
if [ -f "$ENV_UTILS" ]; then
    source "$ENV_UTILS"
else
    detect_device() {
        if command -v npu-smi &> /dev/null || [ -c /dev/davinci0 ]; then echo "npu";
        elif command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then echo "gpu";
        else echo "cpu"; fi
    }
fi

usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Description:
    Wrapper for 'lm_eval' to standardize evaluation on NPU/GPU.
    Supports both 'vllm' (default) and 'hf' backends.

Options:
  --model-path PATH       Path to model (Optional if first arg)
  --backend TYPE          Backend to use: 'vllm' or 'hf' (default: vllm)
  --tasks LIST            Comma-separated tasks (e.g. wikitext,ceval-valid) (default: wikitext)
  --fewshot INT           Number of few-shot examples (default: 0)
  --batch-size SIZE       Batch size or 'auto' (default: auto)
  --output-dir DIR        Directory to save results (default: outputs/eval)
  
  # Hardware/Model Config
  -d, --devices DEV       Devices to use (e.g. "0,1") (default: 0)
  -t, --tp SIZE           Tensor Parallel size (default: 1)
  --gpu-memory UTIL       GPU memory utilization (default: 0.8)
  --max-model-len LEN     Max model length (default: 4096)
  -q, --quantization TYPE Quantization method (e.g., awq, gptq, ascend). 
                          (Auto-set to 'ascend' on NPU if using vllm)

Example:
  bash $0 outputs/qwen-int8 --tasks wikitext --fewshot 5 -d 0,1 -t 2
  bash $0 \
    outputs/compressor/gptq/ascend-qwen3_4b \
    --tasks arc_challenge,arc_easy,boolq,headqa_en,hellaswag,openbookqa,piqa,winogrande \
    -d 1 \
    -q
EOF
}

# --- 默认配置 ---
MODEL_PATH=""
BACKEND="vllm"
TASKS="wikitext"
FEWSHOT=0
BATCH_SIZE="auto"
OUTPUT_DIR="outputs/lmeval"

# 硬件配置
DEVICES="0"
TP_SIZE=1
MEM_UTIL=0.8
MAX_LEN=4096
QUANT_METHOD=""
DEVICE_TYPE=""

POSITIONAL_ARGS=()

# --- 2. 参数解析 ---
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
        --max-model-len) MAX_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="ascend"; shift 1
            fi ;;
        -h|--help) usage; exit 0 ;;
        *) POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# 处理位置参数
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Error: Model path is required."
    usage; exit 1
fi

# --- 3. 环境与参数组装 ---

# 自动检测设备
[[ -z "$DEVICE_TYPE" ]] && DEVICE_TYPE=$(detect_device)

echo "============================================================"
echo "🚀 Starting LM-Evaluation"
echo "   Model:    $MODEL_PATH"
echo "   Backend:  $BACKEND"
echo "   Device:   ${DEVICE_TYPE^^} ($DEVICES)"
echo "   Tasks:    $TASKS"
echo "============================================================"

# 设置环境变量
if [[ "$DEVICE_TYPE" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    export HCCL_OP_EXPANSION_MODE=AIV
    export TASK_QUEUE_ENABLE=1
elif [[ "$DEVICE_TYPE" == "gpu" ]]; then
    export CUDA_VISIBLE_DEVICES=$DEVICES
fi

# 构建 output path
MODEL_NAME=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 创建类似 outputs/eval/model_name/ 结构
SAVE_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
mkdir -p "$SAVE_DIR"
# 输出文件路径 (lmeval 会自动处理 json 后缀，这里不需要加 .json)
OUTPUT_FILE="${SAVE_DIR}/${TASKS//,/_}_${TIMESTAMP}"

# 构建 model_args
MODEL_ARGS="pretrained=${MODEL_PATH},trust_remote_code=True"

if [[ "$BACKEND" == "vllm" ]]; then
    # vLLM 专属参数
    MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
    MODEL_ARGS+=",gpu_memory_utilization=${MEM_UTIL}"
    MODEL_ARGS+=",max_model_len=${MAX_LEN}"
    MODEL_ARGS+=",dtype=auto"
    if [[ -n "$QUANT_METHOD" ]]; then
        MODEL_ARGS+=",quantization=${QUANT_METHOD}"
    fi
elif [[ "$BACKEND" == "hf" ]]; then
    # HuggingFace 专属参数
    # 如果是多卡 HF，通常需要 parallelize=True，但这里简单的用 device map
    if [[ "$TP_SIZE" -gt 1 ]]; then
         MODEL_ARGS+=",parallelize=True"
    fi
fi

echo "🔧 Model Args: $MODEL_ARGS"
echo "📂 Output:     $OUTPUT_FILE"

# --- 4. 执行评测 ---
# 检查环境是否包含必要的库
python -c "import npuslim; import lm_eval" &> /dev/null || {
    echo "❌ Error: 'npuslim' or 'lm-evaluation-harness' not found in current Python environment."
    exit 1
}

echo "🚀 Launching lm_eval with NPUSlim plugin injection..."

PYTHONUNBUFFERED=1 python -u -c "
import sys
import npuslim.plugins as plugin
try:
    plugin.register()
    print('✅ NPUSlim plugin registered successfully.')
except Exception as e:
    print(f'⚠️ Plugin registration failed: {e}')

from lm_eval.__main__ import cli_evaluate
if __name__ == '__main__':
    cli_evaluate()
" \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$FEWSHOT" \
    --output_path "$OUTPUT_FILE"

echo ""
echo "✅ Evaluation completed."
echo "📄 Results saved in: $(dirname "${OUTPUT_FILE}")"
