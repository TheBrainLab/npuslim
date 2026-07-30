#!/bin/bash
# ============================================================
# npuslim offload trunk 推理验证脚本
#
# 用法:
#   bash test_offload_infer.sh              # 使用默认问题
#   bash test_offload_infer.sh "你的问题"    # 自定义问题
# ============================================================

PORT=8199
MODEL_PATH="/home/zzw/llm_infer_workspace/models/Qwen3-30B-A3B"
QUESTION="${1:-请计算 17 乘以 23 等于多少？请直接给出答案。}"

echo "============================================================"
echo "  NPUSlim Offload Trunk 推理验证 (流式)"
echo "============================================================"
echo "  服务地址: http://localhost:$PORT"
echo "  问题:     $QUESTION"
echo "============================================================"
echo ""

# 检查服务是否在线
if ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "❌ vLLM 服务未启动！请先运行 test_offload_serve.sh"
    exit 1
fi

echo "发送流式请求中..."
echo ""
echo "============================================================"
echo "  模型回复 (流式)"
echo "============================================================"
echo ""

# 流式请求：逐字符即时打印
# 使用 --no-buffer 确保不缓冲，python 用 -u 确保无缓冲
FULL_CONTENT=""
TOKEN_COUNT=0

curl -s -N "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_PATH\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$QUESTION\"}],
        \"max_tokens\": 200,
        \"temperature\": 0.1,
        \"stream\": true
    }" | python3 -u -c "
import sys, json

full_content = []
prompt_tokens = None
completion_tokens = None

for line in sys.stdin:
    line = line.strip()
    if not line or not line.startswith('data: '):
        continue
    data = line[6:]
    if data == '[DONE]':
        break
    try:
        chunk = json.loads(data)
        delta = chunk['choices'][0].get('delta', {})
        if 'content' in delta and delta['content']:
            full_content.append(delta['content'])
            print(delta['content'], end='', flush=True)
        usage = chunk.get('usage')
        if usage:
            prompt_tokens = usage.get('prompt_tokens')
            completion_tokens = usage.get('completion_tokens')
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

print()
print()
print(f'--- Token 统计: prompt={prompt_tokens or \"?\"}, completion={completion_tokens or \"?\"} ---')

# 输出完整内容供后续校验
content = ''.join(full_content)
sys.stderr.write(content + '\n')
sys.stderr.flush()
" 2>/tmp/_offload_infer_content

# 读取 stderr 输出的完整内容
FULL_CONTENT=$(cat /tmp/_offload_infer_content)
rm -f /tmp/_offload_infer_content

echo ""
echo "============================================================"
# 简单验证：检查回复是否非空
if [ ${#FULL_CONTENT} -gt 5 ]; then
    echo "  ✅ 推理验证通过：模型返回了有效回复"
else
    echo "  ⚠️  模型回复可能异常，请检查上方输出"
fi
echo "============================================================"
echo ""
echo "查看 offload 运行时日志:"
echo "  grep -i 'offload\|prefetch\|EnhancedNPU\|OffloadMonitor' /tmp/vllm_offload_test.log | tail -20"
