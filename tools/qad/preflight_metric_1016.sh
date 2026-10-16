#!/bin/bash
# =============================================================================
# preflight_metric_1016.sh — verify the mbpp_longctx metric END-TO-END in the
# HEAD container BEFORE launching a long eval (~10 s).
#
# Why (2026-10-16 incident): three runs (~35 min each) died at metric-import
# time after a ~190 s hub timeout, and a follow-up "fix" silently scored every
# sample 0.0.  This guard catches both classes in ~10 s:
#   1) import  -> exercises the metric load (offline / no-hub requirement)
#   2) score a known-correct FENCED solution -> exercises code execution AND
#      the EXTRACT_PATCH_1010B fence extraction, asserting exactly 1.0
#
# env: HEAD_HOST (required), CONTAINER (default e6_noawq_env), TASK_DIR
# exit: 0 = OK ; 1 = metric broken (stderr/stdout explains which part)
# =============================================================================
set -uo pipefail
HEAD_HOST="${HEAD_HOST:?need to set HEAD_HOST}"
CONTAINER="${CONTAINER:-e6_noawq_env}"
TASK_DIR="${TASK_DIR:-/llm_infer_workspace/ylt/glm52-qad/mbpp_longctx}"

ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HEAD_HOST" \
  "docker exec -i -w $TASK_DIR $CONTAINER bash -s" <<'EOS'
export HF_HOME=/llm_infer_workspace/hf_home
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ALLOW_CODE_EVAL=1
python3 -u -c "
import sys
from lm_eval.tasks.mbpp.utils import pass_at_1  # triggers metric load + built-in self-test
fence = chr(96) * 3
gen = fence + 'python\ndef add(a,b):\n    return a+b\n' + fence
score = pass_at_1(['assert add(2,3)==5'], [gen])
print('metric self-test score =', score)
if score != 1.0:
    print('PREFLIGHT FAILED: fenced correct solution scored', score, '(expected 1.0) -> metric broken')
    sys.exit(1)
print('PREFLIGHT OK')
"
EOS
