#!/bin/bash
# Run AISBench (official recipe) inside the container against a local serve.
# Usage: aisbench_run.sh <tag> <port> [<config_suffix>]
set -uo pipefail
TAG="$1"; PORT="${2:-8000}"; CFGSUF="${3:-$TAG}"
AB=/llm_infer_workspace/ylt/glm52-qad/aisbench
CFGDIR=/llm_infer_workspace/ylt/glm52-qad/aisbench_cfgs
cd "$AB"
sed -e "s/HOST_IP_PLACEHOLDER/127.0.0.1/" -e "s/host_port=8000/host_port=${PORT}/" \
    "$CFGDIR/vllm_api_general_chat_${CFGSUF}.py" > "ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat_${TAG}.py"
echo "[aisbench_run] tag=$TAG port=$PORT cfg=$CFGSUF"
python3 -m ais_bench.benchmark.cli.main \
  --models "vllm_api_general_chat_${TAG}" \
  --datasets aime2026_gen_0_shot_chat_prompt gpqa_gen_0_shot_cot_chat_prompt
echo "[aisbench_run] rc=$?"
