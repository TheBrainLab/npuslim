# 模型评估指南

## 1. 评估体系概览

量化模型的评估分为两个维度：

- **模型质量评估**：通过标准基准数据集（如 wikitext、MMLU 等）衡量量化前后模型精度的变化，量化损失应尽量小
- **服务性能评估**：通过压力测试衡量推理吞吐量、延迟等指标，验证量化后模型的部署效率

NPUSlim 提供两套评估工具：

| 工具 | 脚本 | 用途 |
|------|------|------|
| LM-Eval Harness | `tools/eval/run_lmeval.sh` | 模型质量基准评估 |
| Stress Test | `tools/eval/run_stress_test.sh` | 推理性能压力测试 |

## 2. LM-Eval Harness 评估

### 2.1 三种后端模式

`run_lmeval.sh` 支持三种评估后端，适用于不同场景：

#### vllm 后端（推荐，最快）

直接通过 vLLM 加载模型进行评估，无需预先启动服务器。模型加载后直接在进程内推理，速度最快。

```bash
bash tools/eval/run_lmeval.sh outputs/model --backend vllm --tasks wikitext -d 0
```

**适用场景**：本地 GPU/NPU 环境快速评估，仅需推理结果不需要持久化服务。

**注意**：此模式需要 GPU 或 NPU 设备可用。

#### hf 后端

通过 HuggingFace Transformers 原生加载模型进行评估。

```bash
bash tools/eval/run_lmeval.sh outputs/model --backend hf --tasks wikitext -d 0
```

**适用场景**：需要与 HuggingFace 原生推理行为对比的场景。

#### api 后端

通过 OpenAI 兼容 API 调用已部署的 vLLM 服务进行评估。

```bash
# 第一步：部署 vLLM 服务
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# 第二步：运行 API 评估（另一个终端）
bash tools/eval/run_lmeval.sh outputs/model --backend api --tasks wikitext
```

**适用场景**：
- 评估已部署的远程服务
- 需要在独立环境中运行评估
- 评估非本地模型（如云端 API）

**前置条件**：需要先启动 vLLM 服务器，脚本会在评估前自动检查服务连通性和模型身份。

### 2.2 参数详解

#### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | （必填） | 模型路径，支持位置参数 |
| `--backend` | `vllm` | 后端类型：`vllm`、`hf`、`api` |
| `--tasks` | `wikitext` | 评估任务列表，逗号分隔 |
| `--fewshot` | `0` | Few-shot 示例数量 |
| `--batch-size` | `auto` | 批大小，或 `auto` 自动调整 |
| `--output-dir` | `outputs/benchmark/lmeval` | 结果输出目录 |
| `--limit` | 全部 | 每个任务的样本数上限 |
| `--log-samples` | 关闭 | 保存模型输出（用于调试） |
| `--apply-chat-template` | 关闭 | 使用模型自带的聊天模板 |

#### 硬件参数（vllm/hf 后端）

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `-d, --devices` | - | `0` | 设备 ID |
| `-t, --tp` | - | `1` | 张量并行大小 |
| `--max-model-len` | - | `4096` | 最大模型序列长度 |
| `--hccl-port` | - | `60000` | NPU HCCL 基准端口 |
| `--gpu-memory` | - | `0.8` | GPU 显存利用率 |
| `-q, --quantization` | - | 自动 | 量化方法（NPU 自动设为 ascend） |
| `-ep` | - | 关闭 | 启用专家并行（MoE 模型） |
| `--compilation-config` | - | - | 编译配置 |
| `--enforce-eager` | - | 关闭 | 强制即时执行模式 |

#### API 参数（api 后端）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--url` | 自动生成 | API 端点 URL |
| `--port` | `8080` | 服务器端口 |
| `--model-name` | 模型路径 | API 请求中使用的模型名称 |
| `--chat` | 关闭 | 使用 Chat Completions 端点（适用于生成式任务） |

### 2.3 常用评估任务

#### 困惑度评估

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm \
    --tasks wikitext \
    -d 0
```

#### 多任务综合评估

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande \
    -d 0,1 -t 2
```

#### MMLU 评估（生成式）

对 Chat 微调模型使用 MMLU 生成式变体：

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend api \
    --tasks mmlu_generative \
    --chat
```

注意：`--chat` 使用 `/v1/chat/completions` 端点，适用于 `local-chat-completions` 模型。`mmlu`（loglikelihood 版本）与 `--chat` 不兼容，应使用 `mmlu_generative`。

#### 远程 API 评估

```bash
export OPENAI_API_KEY=sk-xxx
bash tools/eval/run_lmeval.sh deepseek-chat \
    --backend api \
    --url https://api.deepseek.com/v1/completions
```

#### 限制样本数快速验证

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm \
    --tasks wikitext \
    --limit 100 \
    -d 0
```

#### 保存模型输出

添加 `--log-samples` 可将每个样本的模型输出保存到结果目录，便于后续分析：

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm \
    --tasks wikitext \
    --log-samples \
    -d 0
```

### 2.4 输出结果

评估结果保存在 `--output-dir` 指定的目录中，文件名格式为 `{任务名}_{时间戳}`。结果包含各项评估指标（如困惑度、准确率等），以 JSON 格式存储。

### 2.5 环境变量

脚本在运行时会自动设置以下环境变量：

| 变量 | 值 | 说明 |
|------|-----|------|
| `OMP_NUM_THREADS` | `1` | 避免 OpenMP 线程池冲突 |
| `MKL_NUM_THREADS` | `1` | 避免 MKL 线程池冲突 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | vLLM 后端使用 spawn 方式创建子进程 |

## 3. 压力测试

### 3.1 概述

压力测试基于 `evalscope perf` 工具，向运行中的 vLLM 服务发送并发请求，测量推理吞吐量和延迟。与 LM-Eval 的质量评估不同，压力测试关注的是服务性能指标。

### 3.2 前置条件

压力测试需要一个**正在运行的 vLLM 服务器**。部署命令：

```bash
# 启动服务
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 --wait

# 等待输出 "vLLM server is ready!" 后，在另一个终端运行压力测试
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | （必填） | 模型路径（用于 tokenizer） |
| `--model-name` | 模型路径 | API 请求中使用的模型名称 |
| `--url` | 自动生成 | 端点 URL |
| `--port` | `8080` | 服务器端口 |
| `--parallel` | `"1 10 50 100"` | 并发级别列表 |
| `--total-requests` | `"10 50 100 200"` | 各级别请求数 |
| `--prompt-length` | `1024` | 提示词长度（token 数） |
| `--max-tokens` | `1024` | 最大生成 token 数 |
| `--output-dir` | `outputs/benchmark/stress_test` | 结果输出目录 |

### 3.4 使用示例

#### 基本压力测试

```bash
bash tools/eval/run_stress_test.sh outputs/model
```

使用默认配置（并发 1/10/50/100，请求数 10/50/100/200）对本地服务进行测试。

#### 自定义并发与请求数

```bash
bash tools/eval/run_stress_test.sh outputs/model \
    --parallel "1 16 32" \
    --total-requests "20 100 200"
```

`--parallel` 和 `--total-requests` 为一一对应关系，即：
- 并发 1，发送 20 个请求
- 并发 16，发送 100 个请求
- 并发 32，发送 200 个请求

#### 自定义序列长度

```bash
bash tools/eval/run_stress_test.sh outputs/model \
    --prompt-length 2048 \
    --max-tokens 512
```

#### 远程服务压力测试

```bash
export OPENAI_API_KEY=sk-xxx
bash tools/eval/run_stress_test.sh deepseek-chat \
    --url https://api.deepseek.com/v1/chat/completions \
    --model-name deepseek-chat
```

### 3.5 结果解读

测试完成后，`evalscope` 会输出以下关键指标：

| 指标 | 说明 |
|------|------|
| **Throughput** | 每秒处理的请求数（requests/s）或 token 数（tokens/s） |
| **Latency (mean)** | 平均请求延迟 |
| **Latency (P50/P90/P99)** | 延迟分位数 |
| **TTFT** | 首 token 延迟（Time To First Token） |
| **ITL** | token 间延迟（Inter-Token Latency） |

结果保存在 `--output-dir` 目录中，同时会在终端实时显示。

### 3.6 测试前健康检查

脚本在执行前会自动进行两项检查：

1. **服务连通性**：访问 `/health` 端点确认服务在线
2. **模型验证**：通过 `/v1/models` 端点确认加载的模型与预期一致

若检查失败，脚本会提示先部署服务：

```
Tip: Deploy server first: bash tools/serve/deploy_vllm.sh <model> -d 0 -t 1
```

## 4. 评估最佳实践

### 4.1 推荐评估流程

```
量化模型 → 质量评估 → 部署服务 → 性能评估 → 对比分析
```

具体步骤：

#### 第一步：量化

```bash
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

#### 第二步：质量评估（vllm 后端，无需部署）

```bash
# 困惑度评估
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm --tasks wikitext -d 0

# 综合基准评估
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm \
    --tasks arc_challenge,hellaswag,piqa,winogrande \
    -d 0
```

#### 第三步：部署服务

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 --wait
```

#### 第四步：性能压力测试

```bash
bash tools/eval/run_stress_test.sh outputs/model
```

### 4.2 对比基线建议

为评估量化的影响，建议对原始模型和量化模型分别运行相同的评估任务，然后对比指标：

```bash
# 原始模型
bash tools/eval/run_lmeval.sh Qwen/Qwen3-8B \
    --backend vllm --tasks wikitext -d 0

# 量化模型
bash tools/eval/run_lmeval.sh outputs/qwen3-int8 \
    --backend vllm --tasks wikitext -d 0
```

### 4.3 NPU 环境评估注意事项

- NPU 评估需使用 `-q` 参数指定 Ascend 量化方法
- 多卡 NPU 环境下需配置 `--hccl-port` 避免端口冲突
- 如遇到图捕获问题，添加 `--enforce-eager` 参数

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend vllm --tasks wikitext \
    -d 0 -q --enforce-eager
```

### 4.4 离线评估

在网络受限环境中，添加 `--offline` 参数使用本地缓存的数据集和模型：

```bash
bash tools/eval/run_lmeval.sh outputs/model \
    --backend hf --tasks wikitext \
    --offline -d 0
```

### 4.5 大规模评估建议

对于包含多个任务的完整评估，建议：

- 使用 `--output-dir` 指定独立的输出目录以便管理
- 使用 `--log-samples` 保存样本输出，方便后续排查
- 合理设置 `--batch-size`，过大会导致 OOM，过小会降低效率
- 对于首次运行，先用 `--limit 100` 快速验证流程，再进行全量评估
