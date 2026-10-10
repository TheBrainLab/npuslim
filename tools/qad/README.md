# tools/qad — QAD 流水线核心脚本（仓库快照）

> ⚠️ 快照性质：本目录是 C3 共享工作目录 `ylt/glm52-qad/` 的**快照**，用于仓内复现/存档。
> **不在本机直接运行**：脚本用 `sys.path.insert(0, "/llm_infer_workspace/ylt/npuslim/src")`
> 绝对路径注入 import（仅容器内该路径存在），并引用共享数据/产物路径。

## 必需 env（多数旋钮已 env 化）
GLM_TRAIN / OFFLOAD_PARAM / QAD_FLUSH_MODE / QAD_SLAB / QAD_EP / QAD_EP_TMAX /
GLM_DEPLOY_SCALE_DIR / QAD_TPROF(+_START/_ACTIVE/_OUT) / PYTORCH_NPU_ALLOC_CONF

## 待办（Phase D，见方案 §3.7/S16）
路径参数化：`sys.path.insert(<abs>)` 与硬编码数据/产物路径 → env（NPUSLIM_SRC / QAD_ROOT，带默认回退）。
