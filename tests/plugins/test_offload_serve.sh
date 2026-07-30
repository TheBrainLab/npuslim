#!/bin/bash
# ============================================================
# npuslim offload trunk 端到端验证启动脚本
#
# 用法:
#   bash test_offload_serve.sh              # 启动服务（自动计算 offload 量）
#   bash test_offload_serve.sh --no-offload # 不启用 offload（对照组）
#
# 环境变量 (全部有默认值，按需覆盖):
#
#   === 基本配置 ===
#   MODEL_PATH              模型路径
#   PORT                    服务端口
#   TP_SIZE                 张量并行度 (默认 1)
#   GPU_MEM_UTIL            GPU 内存利用率 (默认 0.85)
#   MAX_MODEL_LEN           最大序列长度 (默认 4096)
#
#   === Offload Trunk 配置 ===
#   NPUSLIM_OFFLOAD_STRATEGY       策略: size_aware / group / custom (默认 size_aware)
#   NPUSLIM_OFFLOAD_GROUP_SIZE     group 策略参数 (默认 0)
#   NPUSLIM_OFFLOAD_NUM_IN_GROUP   group 策略参数 (默认 1)
#   NPUSLIM_OFFLOAD_PREFETCH_STEP  预取步数上限 (默认 1，实际自动适配)
#   NPUSLIM_OFFLOAD_PATTERNS       custom 策略 offload pattern, 逗号分隔
#   NPUSLIM_OFFLOAD_KEEP_PATTERNS  custom 策略 keep pattern, 逗号分隔
#   NPUSLIM_OFFLOAD_PARAMS         参数级过滤, 逗号分隔 (如 w2_weight,gate_up_weight)
#   NPUSLIM_OFFLOAD_SAFETY_MARGIN_GB   安全余量 GB (默认 2)
#   NPUSLIM_OFFLOAD_CPU_MEM_THRESHOLD  CPU 内存阈值 (默认 0.6)
#
#   NPUSLIM_OFFLOAD_STRICT_CHECK    1=内存不足报错中止, 0=只警告 (默认 1)
#   NPUSLIM_OFFLOAD_ENABLE_MONITOR  1=开启运行时监控 (默认 1)
#
#   注意: offload 量由程序自动计算，无需手动指定。
#   KV cache 大小根据模型结构和 max_model_len/max_num_seqs 精确估算。
#   激活值和图编译开销由 vllm 的 gpu_memory_utilization 自动管理。
#
# 示例:
#   # 基本用法: 自动计算 offload
#   bash test_offload_serve.sh
#
#   # 开启 trace 查看 prefetch 动态过程
#   NPUSLIM_OFFLOAD_TRACE=1 bash test_offload_serve.sh
#
#   # 使用 group 策略 (每 4 层 offload 1 层)
#   NPUSLIM_OFFLOAD_STRATEGY=group NPUSLIM_OFFLOAD_GROUP_SIZE=4 bash test_offload_serve.sh
#
#   # 对照组: 不启用 offload
#   bash test_offload_serve.sh --no-offload
#
#   # 只 offload w2_weight 参数
#   NPUSLIM_OFFLOAD_PARAMS=w2_weight bash test_offload_serve.sh
# ============================================================

set -e

# === 基本配置 ===
MODEL_PATH="${MODEL_PATH:-/home/zzw/llm_infer_workspace/models/Qwen3-30B-A3B}"
PORT="${PORT:-8199}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
LOG_FILE="/tmp/vllm_offload_test.log"

# === 参数 ===
DISABLE_OFFLOAD=false
if [ "$1" = "--no-offload" ] || [ "$1" = "0" ]; then
    DISABLE_OFFLOAD=true
fi

# === Offload Trunk 配置 ===
STRATEGY="${NPUSLIM_OFFLOAD_STRATEGY:-size_aware}"
GROUP_SIZE="${NPUSLIM_OFFLOAD_GROUP_SIZE:-0}"
NUM_IN_GROUP="${NPUSLIM_OFFLOAD_NUM_IN_GROUP:-1}"
PREFETCH_STEP="${NPUSLIM_OFFLOAD_PREFETCH_STEP:-1}"
OFFLOAD_PATTERNS="${NPUSLIM_OFFLOAD_PATTERNS:-}"
KEEP_PATTERNS="${NPUSLIM_OFFLOAD_KEEP_PATTERNS:-}"
OFFLOAD_PARAMS="${NPUSLIM_OFFLOAD_PARAMS:-}"
SAFETY_MARGIN_GB="${NPUSLIM_OFFLOAD_SAFETY_MARGIN_GB:-2}"
CPU_MEM_THRESHOLD="${NPUSLIM_OFFLOAD_CPU_MEM_THRESHOLD:-0.6}"

# === 配置 ===
TRACE="${NPUSLIM_OFFLOAD_TRACE:-0}"
STRICT_CHECK="${NPUSLIM_OFFLOAD_STRICT_CHECK:-1}"
ENABLE_MONITOR="${NPUSLIM_OFFLOAD_ENABLE_MONITOR:-1}"

# 清理可能残留的进程
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "api_server" 2>/dev/null || true
sleep 2

echo "============================================================"
echo "  NPUSlim Offload Trunk 端到端验证"
echo "============================================================"
echo "  模型:           $MODEL_PATH"
echo "  端口:           $PORT"
echo "  张量并行:        $TP_SIZE"
echo "  GPU内存利用率:   $GPU_MEM_UTIL"
echo "  最大序列长度:    $MAX_MODEL_LEN"
echo "  ────────────────────────────────────────"
echo "  Offload:         $([ "$DISABLE_OFFLOAD" = "false" ] && echo "启用 (自动计算)" || echo "禁用 (对照组)")"
echo "  策略:            $STRATEGY"
echo "  Prefetch步数上限: $PREFETCH_STEP"
echo "  安全余量:        ${SAFETY_MARGIN_GB}GB"
echo "  CPU内存阈值:     $CPU_MEM_THRESHOLD"
if [ -n "$OFFLOAD_PATTERNS" ]; then
echo "  Offload patterns: $OFFLOAD_PATTERNS"
fi
if [ -n "$KEEP_PATTERNS" ]; then
echo "  Keep patterns:    $KEEP_PATTERNS"
fi
if [ -n "$OFFLOAD_PARAMS" ]; then
echo "  参数过滤:        $OFFLOAD_PARAMS"
fi
if [ "$STRATEGY" = "group" ]; then
echo "  Group size:      $GROUP_SIZE"
echo "  Num in group:    $NUM_IN_GROUP"
fi
echo "  ────────────────────────────────────────"
echo "  Trace日志:       $TRACE (0=关闭/compile, 1=开启/eager)"
echo "  严格内存检查:    $STRICT_CHECK"
echo "  运行时监控:      $ENABLE_MONITOR"
echo "  ────────────────────────────────────────"
echo "  日志文件:        $LOG_FILE"
echo "============================================================"
echo ""

# 必须开启 NPUSLIM_PLUGIN_ENABLE 才能让 npuslim patch 生效
export NPUSLIM_PLUGIN_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

if [ "$DISABLE_OFFLOAD" = "true" ]; then
    echo "[模式] 不启用 offload trunk（对照组）"
    VLLM_CMD="vllm serve \"$MODEL_PATH\" \
        --tensor-parallel-size $TP_SIZE \
        --trust-remote-code \
        --port $PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN"
else
    # trace 开启时必须用 --enforce-eager
    EXTRA_ARGS=""
    if [ "$TRACE" = "1" ]; then
        echo "[模式] offload trunk + trace=ON + eager mode（调试用）"
        EXTRA_ARGS="--enforce-eager"
    else
        echo "[模式] offload trunk + trace=OFF + compile mode（性能优先）"
    fi

    # 构建 additional_config JSON
    CONFIG_JSON="{\"npuslim_offload_trunk\": {"
    CONFIG_JSON+="\"enabled\": true"
    CONFIG_JSON+=", \"strategy\": \"$STRATEGY\""
    CONFIG_JSON+=", \"prefetch_step\": $PREFETCH_STEP"
    CONFIG_JSON+=", \"safety_margin_gb\": $SAFETY_MARGIN_GB"
    CONFIG_JSON+=", \"cpu_memory_threshold\": $CPU_MEM_THRESHOLD"
    CONFIG_JSON+=", \"strict_memory_check\": $([ "$STRICT_CHECK" = "1" ] && echo true || echo false)"
    CONFIG_JSON+=", \"enable_monitor\": $([ "$ENABLE_MONITOR" = "1" ] && echo true || echo false)"

    if [ "$STRATEGY" = "group" ]; then
        CONFIG_JSON+=", \"group_size\": $GROUP_SIZE"
        CONFIG_JSON+=", \"num_in_group\": $NUM_IN_GROUP"
    fi

    if [ -n "$OFFLOAD_PATTERNS" ]; then
        PATTERNS_JSON=$(echo "$OFFLOAD_PATTERNS" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')
        CONFIG_JSON+=", \"offload_layer_patterns\": [$PATTERNS_JSON]"
    fi

    if [ -n "$KEEP_PATTERNS" ]; then
        KEEP_JSON=$(echo "$KEEP_PATTERNS" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')
        CONFIG_JSON+=", \"keep_layer_patterns\": [$KEEP_JSON]"
    fi

    if [ -n "$OFFLOAD_PARAMS" ]; then
        PARAMS_JSON=$(echo "$OFFLOAD_PARAMS" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')
        CONFIG_JSON+=", \"offload_params\": [$PARAMS_JSON]"
    fi

    CONFIG_JSON+="}}"

    echo "[配置] additional-config: $CONFIG_JSON"
    echo ""

    VLLM_CMD="vllm serve \"$MODEL_PATH\" \
        --tensor-parallel-size $TP_SIZE \
        --trust-remote-code \
        --port $PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        $EXTRA_ARGS \
        --additional-config '$CONFIG_JSON'"
fi

# 启动 vllm
eval "$VLLM_CMD > \"$LOG_FILE\" 2>&1 &"

VLLM_PID=$!
echo "vLLM 服务启动中 (PID=$VLLM_PID)..."
echo "等待服务就绪 (最多等待 5 分钟)..."
echo ""

# 等待服务就绪
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "============================================================"
        echo "  ✅ vLLM 服务已就绪！"
        echo "  服务地址: http://localhost:$PORT"
        echo "  日志文件: $LOG_FILE"
        echo "============================================================"
        echo ""
        echo "常用日志查看命令:"
        echo "  grep 'OffloadTrunk' $LOG_FILE"
        echo "  grep '\[TRACE\]' $LOG_FILE"
        echo "  grep '内存' $LOG_FILE"
        echo "  grep 'KV cache' $LOG_FILE"
        echo ""
        echo "运行推理测试:"
        echo "  bash /home/zzw/llm_infer_workspace/zzw/code/vllm-workspace/npuslim/tests/plugins/test_offload_infer.sh"
        echo ""
        echo "停止服务: kill $VLLM_PID"
        echo ""
        exit 0
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "❌ vLLM 服务启动失败！"
        echo "最近 50 行日志:"
        tail -50 "$LOG_FILE"
        exit 1
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "❌ vLLM 服务启动超时！"
tail -50 "$LOG_FILE"
exit 1
