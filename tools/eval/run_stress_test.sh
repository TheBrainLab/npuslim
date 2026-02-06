#!/bin/bash

# --- 1. 帮助与配置 ---
usage() {
    cat << EOF
Usage: $0 [MODEL_PATH] [OPTIONS]

Description:
    Automated Benchmark Pipeline:
    1. Deploys vLLM server (using tools/deploy/deploy_vllm.sh)
    2. Waits for service readiness (Infinite wait until ready or Ctrl+C)
    3. Runs stress test (using tools/eval/evalscope_perf.sh)
    4. Cleans up processes automatically

Options:
  --model-path PATH       Path to model (Optional if first arg)
  --port PORT             Service port (default: 8080)
  --outputs-dir DIR       Directory for logs and results (default: outputs/benchmark/stress_test)
  
  # Deploy Options
  -d, --devices DEV       Devices to use (e.g., "0,1") (default: 0)
  -t, --tp SIZE           Tensor Parallel size (default: 1)
  --gpu-memory UTIL       GPU memory utilization (default: 0.9)
  --max-model-len LEN     Max model length (default: 4096)

  # Stress Test Options (Pass-through)
  --parallel LIST         Concurrency levels (default: "1 10 50 100")
  --total-requests LIST   Total requests per level (default: "10 50 100 200")
  
  -h, --help              Show this message

Example:
  bash $0 outputs/qwen-int8 -t 2 --parallel "1 100" --total-requests "20 200"
EOF
}

# 脚本路径定义
DEPLOY_SCRIPT="tools/deploy/deploy_vllm.sh"
STRESS_SCRIPT="tools/eval/evalscope_perf.sh"

# --- 默认参数 ---
MODEL_PATH=""
PORT=8080
DEVICES="0"
TP_SIZE=1
MEM_UTIL=0.9
MAX_LEN=4096
OUTPUTS_DIR="outputs/benchmark/stress_test"

# 默认压测配置 (如果不传则用这个)
PARALLEL_CONFIG="1 10 50 100"
REQUEST_CONFIG="10 50 100 200"

POSITIONAL_ARGS=()

# --- 2. 参数解析 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        # 基础配置
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
        
        # 部署配置
        -d|--devices) DEVICES="$2"; shift 2 ;;
        -t|--tp|--tensor-parallel) TP_SIZE="$2"; shift 2 ;;
        --gpu-memory) MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_LEN="$2"; shift 2 ;;

        # 压测配置 (新增透传)
        --parallel) PARALLEL_CONFIG="$2"; shift 2 ;;
        --number|--total-requests) REQUEST_CONFIG="$2"; shift 2 ;;

        -h|--help) usage; exit 0 ;;
        *) POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# 处理位置参数
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

# 校验
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Error: Model path is required."
    usage
    exit 1
fi

if [[ ! -f "$DEPLOY_SCRIPT" || ! -f "$STRESS_SCRIPT" ]]; then
    echo "❌ Error: Core scripts not found."
    echo "   Checked: $DEPLOY_SCRIPT"
    echo "   Checked: $STRESS_SCRIPT"
    echo "   Please run this script from the project root."
    exit 1
fi

# --- 3. 定义强力清理陷阱 (Trap) ---
cleanup() {
    echo ""
    echo "🛑 [Pipeline] Interrupted! Cleaning up..."
    
    if [[ -n "$SERVER_PID" ]]; then
        # 1. 查找子进程 (vLLM Python Process)
        CHILD_PIDS=$(pgrep -P $SERVER_PID)
        
        echo "   -> Killing vLLM Server Wrapper (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        
        if [[ -n "$CHILD_PIDS" ]]; then
             echo "   -> Killing vLLM Python Processes (PIDs: $CHILD_PIDS)..."
             kill $CHILD_PIDS 2>/dev/null
             
             # Double Check: 强力击杀
             sleep 2
             kill -9 $CHILD_PIDS 2>/dev/null
        fi
        
        # 兜底：直接按端口杀，防止有漏网之鱼
        echo "   -> Ensuring port $PORT is free..."
        lsof -t -i:$PORT | xargs -r kill -9 2>/dev/null
    fi
    echo "✨ Done."
}
trap cleanup EXIT INT TERM

# --- 4. 阶段一：启动服务 ---
mkdir -p "$OUTPUTS_DIR"
SERVER_LOG="${OUTPUTS_DIR}/server_startup.log"

echo "============================================================"
echo "🚀 [Stage 1] Deploying vLLM Server..."
echo "   Model:    $MODEL_PATH"
echo "   Port:     $PORT"
echo "   Devices:  $DEVICES"
echo "   Log:      $SERVER_LOG"
echo "============================================================"
echo "💡 Tip: Monitor logs with: tail -f $SERVER_LOG"
echo "============================================================"

# 后台启动部署脚本
bash "$DEPLOY_SCRIPT" \
    "$MODEL_PATH" \
    --port "$PORT" \
    --devices "$DEVICES" \
    --tensor-parallel "$TP_SIZE" \
    --gpu-memory "$MEM_UTIL" \
    --max-model-len "$MAX_LEN" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "⏳ Server process launched (PID: $SERVER_PID). Checking Health..."

# --- 5. 阶段二：健康检查 ---
START_TIME=$(date +%s)

while true; do
    # 进程存活检查
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "❌ [Fatal] Server process died unexpectedly! Check logs: $SERVER_LOG"
        echo "----------------- Log Snippet -----------------"
        tail -n 20 "$SERVER_LOG"
        exit 1
    fi

    # HTTP 状态码检查
    HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" "http://localhost:${PORT}/health")

    if [[ "$HTTP_CODE" == "200" ]]; then
        echo ""
        echo "✅ [Stage 1] Service is UP and READY (HTTP 200)!"
        break
    fi

    # 动态提示
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo -ne "   ⏳ Loading... (${ELAPSED}s) | Press [Ctrl+C] to abort | Log: $SERVER_LOG \033[K\r"
    sleep 3
done

# --- 6. 阶段三：运行压测 ---
echo ""
echo "============================================================"
echo "🔥 [Stage 2] Running Stress Test..."
echo "   Config: Parallel=[$PARALLEL_CONFIG]"
echo "           Requests=[$REQUEST_CONFIG]"
echo "============================================================"

# 调用压测脚本 (透传参数)
bash "$STRESS_SCRIPT" \
    "$MODEL_PATH" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --outputs-dir "$OUTPUTS_DIR" \
    --parallel "$PARALLEL_CONFIG" \
    --total-requests "$REQUEST_CONFIG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ [Pipeline] Benchmark completed successfully!"
    echo "📂 Results saved to: $OUTPUTS_DIR"
else
    echo "⚠️ [Pipeline] Benchmark finished with errors (Code: $EXIT_CODE)."
fi

# 脚本退出时会自动触发 cleanup