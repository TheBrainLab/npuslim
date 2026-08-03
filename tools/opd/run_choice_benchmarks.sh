#!/usr/bin/env bash
set -euo pipefail

# Run the seven choice-style OPD benchmark tasks with NPUSlim's lm-eval wrapper.
#
# Usage:
#   bash tools/opd/run_choice_benchmarks.sh /path/to/model --backend vllm -d 0 -t 1 -q ascend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${1:-}"
if [[ -z "${MODEL_PATH}" || "${MODEL_PATH}" == "-h" || "${MODEL_PATH}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash tools/opd/run_choice_benchmarks.sh MODEL_PATH [run_lmeval options]

Tasks:
  arc_challenge,arc_easy,boolq,headqa_en,openbookqa,piqa,winogrande

Examples:
  bash tools/opd/run_choice_benchmarks.sh /models/qwen3-30b-bf16 --backend vllm -d 0,1 -t 2
  bash tools/opd/run_choice_benchmarks.sh /models/qwen3-4b-w4a16 --backend vllm -d 0 -t 1 -q ascend
EOF
  exit 0
fi
shift

TASKS="arc_challenge,arc_easy,boolq,headqa_en,openbookqa,piqa,winogrande"
OUTPUT_DIR="${OPD_LMEVAL_OUTPUT_DIR:-outputs/opd/lmeval}"

bash "${PROJECT_ROOT}/tools/eval/run_lmeval.sh" "${MODEL_PATH}" \
  --tasks "${TASKS}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-samples \
  "$@"
