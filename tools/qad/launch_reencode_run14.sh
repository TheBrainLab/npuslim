#!/bin/bash
# launch_reencode_run14.sh — 在容器内执行 reencode（deploy-compose，一步到位）
cd /llm_infer_workspace/ylt/glm52-qad && \
mkdir -p reencode_run14_step600 && \
PYTHONUNBUFFERED=1 python3 -u reencode_qad_deliverable.py \
  --artifact /llm_infer_workspace/ylt/npuslim/outputs_npu/glm5/gptq/glm5-w4a8_replica_l0_clip_lo05_omni \
  --lora /llm_infer_workspace/ylt/glm52-qad/ckpt_e2/lora_step600_p.pt \
  --out /llm_infer_workspace/ylt/glm52-qad/reencode_run14_step600 \
  --scaling 4.0 \
  --shard-gb 8.0 \
  > reencode_run14_step600/reencode_run14_step600.out 2>&1
