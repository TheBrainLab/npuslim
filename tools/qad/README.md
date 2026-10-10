# tools/qad — QAD 流水线核心脚本（仓库快照）

> **快照性质**：本目录是 C3 共享工作目录 `ylt/glm52-qad/` 的**快照**，用于仓内复现/存档。

## 路径参数化（Phase D，已实施）

脚本**不再硬编码**绝对路径——统一由 env 驱动，**默认值即历史路径**（零行为变化）：

| env | 默认 | 含义 |
|---|---|---|
| `SHARED` | `/llm_infer_workspace` | 容器内挂载根 |
| `SHARED_HOST` | `/home/yult/llm_infer_workspace` | 宿主机挂载根（launcher 用）|
| `YLT` | `$SHARED/ylt` | 工作根（= 宿主 `/home/yult/llm_infer_workspace/ylt`）|
| `NPUSLIM_SRC` | `$YLT/npuslim/src` | 代码（`sys.path.insert` 目标）|
| `QAD_ROOT` | `$YLT/glm52-qad` | QAD 运行目录（数据/ckpt）|

> `/llm_infer_workspace`（容器）与 `/home/yult/llm_infer_workspace`（宿主）是**同一块盘**；
> `/llm_infer_workspace/ylt` = `/home/yult/llm_infer_workspace/ylt`。
> 覆盖示例：`SHARED=/opt/x NPUSLIM_SRC=/opt/x/ylt/npuslim/src`（可在任意目录运行）。

## 必需 env（多数旋钮已 env 化）

GLM_TRAIN / OFFLOAD_PARAM / QAD_FLUSH_MODE / QAD_SLAB / QAD_EP / QAD_EP_TMAX /
GLM_DEPLOY_SCALE_DIR / QAD_TPROF(+_START/_ACTIVE/_OUT) / PYTORCH_NPU_ALLOC_CONF

## 运行前提

- **import 依赖**：容器内 `$NPUSLIM_SRC` 存在（默认 `/llm_infer_workspace/ylt/npuslim/src`）；
- **数据/产物**：`$QAD_ROOT`（默认 `/llm_infer_workspace/ylt/glm52-qad`）与 `$SHARED/models/...` 存在；
- 宿主侧 launcher（`diag_launch.sh` 等）走 `$SHARED_HOST`。
