#!/bin/bash
# Run AISBench (official recipe) inside the container against a local serve.
# 1019 decision: AIME2026 only (full 30, official recipe). GPQA = over-budget
# (official recipe ~6h/arm due to temp 0.6 long generations) -> skipped/registered.
# Usage: aisbench_run.sh <tag> <port> [<config_suffix>]
set -uo pipefail
TAG="$1"; PORT="${2:-8000}"; CFGSUF="${3:-$TAG}"; DATASETS="${4:-aime2026_gen_0_shot_chat_prompt}"
AB=/llm_infer_workspace/ylt/glm52-qad/aisbench
CFGDIR=/llm_infer_workspace/ylt/glm52-qad/aisbench_cfgs
OUT=/llm_infer_workspace/ylt/glm52-qad/audit1009
cd "$AB"
sed -e "s/HOST_IP_PLACEHOLDER/127.0.0.1/" -e "s/host_port=8000/host_port=${PORT}/" \
    "$CFGDIR/vllm_api_general_chat_${CFGSUF}.py" > "ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat_${TAG}.py"
echo "[aisbench_run] tag=$TAG port=$PORT cfg=$CFGSUF datasets=$DATASETS"
python3 -m ais_bench.benchmark.cli.main \
  --models "vllm_api_general_chat_${TAG}" \
  --datasets ${DATASETS} \
  -w "outputs_${TAG}"
echo "[aisbench_run] rc=$?"
EXP=$(ls -td "outputs_${TAG}"/*/ 2>/dev/null | head -1)
echo "[aisbench_run] exp=$EXP"
# NOTE: no archiving here. This script runs as root inside the container; dirs
# it creates under $OUT would be root-owned on the host and unwritable by yult.
# Archiving is done host-side by the chains (after this returns).
