# 模型部署指南

## 1. vLLM 部署概览

NPUSlim 量化后的模型可通过 vLLM 推理框架进行高效部署。vLLM 提供了 OpenAI 兼容的 API 服务接口，支持 GPU 和 NPU（Ascend）两种硬件平台。部署流程为：

```
量化模型输出 → vLLM 服务启动 → OpenAI 兼容 API → 客户端调用
```

部署脚本 `tools/serve/deploy_vllm.sh` 封装了环境配置、设备检测、参数传递和服务健康检查等完整流程，支持一键启动 vLLM 推理服务。

## 2. deploy_vllm.sh 脚本详解

### 2.1 脚本位置

```bash
tools/serve/deploy_vllm.sh
```

### 2.2 完整参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_PATH` | - | （必填） | 模型路径，支持位置参数或 `--model-path` 指定 |
| `-d, --devices` | - | `0,1` | 设备 ID，多卡用逗号分隔 |
| `-t, --tp` | - | `2` | 张量并行大小（Tensor Parallel） |
| `-p, --pp` | - | `1` | 流水线并行大小（Pipeline Parallel） |
| `--port` | - | `8080` | 服务监听端口 |
| `--hccl-port` | - | `60000` | NPU 通信 HCCL 基准端口 |
| `--gpu-memory` | `-g` | `0.8` | GPU 显存利用率（0-1 之间） |
| `--max-model-len` | - | `4096` | 最大模型序列长度 |
| `-q, --quantization` | - | 自动检测 | 量化方法，NPU 环境自动设置为 `ascend` |
| `--compilation-config` | - | - | 编译配置，如 `'{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| `--media-path` | - | - | 多模态模型允许的本地媒体路径 |
| `-ep, --enable-expert-parallel` | - | 关闭 | 启用 MoE 模型的专家并行 |
| `--enforce-eager` | - | 关闭 | 强制使用即时执行模式，禁用图捕获 |
| `--log-dir` | - | `logs/vllm_server` | 日志保存目录 |
| `--no-log` | - | 关闭 | 禁用日志文件输出 |
| `--wait` | - | 关闭 | 阻塞等待服务就绪 |
| `-h, --help` | - | - | 显示帮助信息 |

### 2.3 GPU 部署流程

#### 基本部署（单卡）

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1
```

#### 多卡张量并行

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0,1 -t 2 --port 8080
```

#### 等待服务就绪

添加 `--wait` 参数，脚本将阻塞直到服务健康检查通过：

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0,1 -t 2 --wait
```

#### 自定义显存利用率与序列长度

```bash
bash tools/serve/deploy_vllm.sh outputs/model \
    -d 0,1 -t 2 \
    --gpu-memory 0.9 \
    --max-model-len 8192
```

GPU 环境下脚本会自动设置 `CUDA_VISIBLE_DEVICES` 环境变量，无需手动配置。

### 2.4 NPU/Ascend 部署流程

#### 基本 NPU 部署

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 -q
```

`-q` 参数（无值）在 NPU 环境下自动选择 `ascend` 量化方法。脚本内部通过检测 `npu-smi` 命令或 `/dev/davinci0` 设备节点来自动识别 NPU 平台。

#### NPU 多卡并行

```bash
bash tools/serve/deploy_vllm.sh outputs/model \
    -d 0,1 -t 2 \
    --hccl-port 60000 \
    -q
```

#### NPU 环境变量说明

脚本在 NPU 模式下自动设置以下关键环境变量：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | 由 `-d` 参数指定 | NPU 可见设备 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:False` | NPU 内存分配策略 |
| `HCCL_IF_BASE_PORT` | `60000` | HCCL 通信基准端口 |
| `HCCL_BUFFSIZE` | `512` | HCCL 缓冲区大小 |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 算子扩展模式 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | TP>1 时启用 | 多卡 FlashComm1 加速 |

#### NPU 部署注意事项

- 若 HCCL 端口绑定失败，使用 `--hccl-port` 指定其他端口
- 若图捕获失败，尝试 `--enforce-eager` 或 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
- MoE 模型建议启用 `-ep` 以开启专家并行

### 2.5 MoE 模型部署

MoE（混合专家）模型需要启用专家并行以提高推理效率：

```bash
bash tools/serve/deploy_vllm.sh outputs/moe_model \
    -d 0,1 -t 2 \
    -ep \
    -q
```

当张量并行大于 1 且未启用 `-ep` 时，脚本会输出提示建议开启专家并行。

## 3. GPTQ GPU 到 NPU 转换

### 3.1 使用场景

GPTQ 量化模型在 GPU 上的权重格式（行方向打包）与 NPU/Ascend 要求的格式（列方向打包）不同。`gptq_gpu_to_npu.py` 工具负责完成格式转换：

- **输入**：标准 GPTQ 量化模型（GPU 格式），包含 `qweight`、`qzeros`、`scales`、`g_idx`
- **输出**：Ascend NPU 格式模型，包含 `weight`、`weight_scale`、`weight_offset`、`quant_model_description.json`

### 3.2 转换命令

```bash
python tools/convert/gptq_gpu_to_npu.py \
    --input /path/to/gpu_gptq_model \
    --output /path/to/npu_model
```

### 3.3 详细参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | （必填） | GPU 格式 GPTQ 模型目录 |
| `--output` | `-o` | （必填） | 输出 NPU 格式模型目录 |
| `--device` | - | `cpu` | 转换使用的设备（推荐 CPU） |
| `--verbose` | `-v` | 关闭 | 启用详细日志 |

### 3.4 转换流程说明

1. 读取 `config.json` 中的 `quantization_config`（bits、group_size 等）
2. 加载 GPTQ 模型权重
3. 解包 int32 压缩权重为独立的 4-bit 值
4. 将无符号表示 `[0, 15]` 转换为有符号表示 `[-8, 7]`
5. 按 Ascend 列方向格式重新打包
6. 转换 scales/zeros 为 `weight_scale`/`weight_offset` 格式
7. 移除 `quantization_config`（防止 HF/vLLM 按 GPU GPTQ 格式加载）
8. 生成 `quant_model_description.json` 供 vLLM-Ascend 使用

### 3.5 限制

- 仅支持 4-bit 量化（W4A16）的转换
- 转换过程在 CPU 上执行（推荐），适合大模型场景
- 输出模型的 `config.json` 中会添加 `ascend_quant_config`

## 4. 部署验证

### 4.1 健康检查

服务启动后，可通过 HTTP 请求验证状态：

```bash
# 基本健康检查
curl http://localhost:8080/health

# 预期返回
# {"status": "ok"} 或纯文本 "ok"
```

### 4.2 模型列表验证

```bash
curl http://localhost:8080/v1/models
```

返回当前服务加载的模型信息，包括模型 ID。

### 4.3 推理测试

使用 OpenAI 兼容的 Completions API 进行简单推理验证：

```bash
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "outputs/model",
        "prompt": "Hello, world!",
        "max_tokens": 32
    }'
```

或使用 Chat Completions API：

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "outputs/model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 32
    }'
```

### 4.4 使用 --wait 自动验证

部署时加 `--wait` 参数，脚本会自动循环检查服务健康状态直到就绪：

```bash
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 --wait
```

## 5. 常见问题排查

### 5.1 NPU HCCL 通信失败

**症状**：多卡 NPU 部署时出现 HCCL 端口绑定或通信超时错误。

**解决方案**：
```bash
# 更换 HCCL 基准端口
bash tools/serve/deploy_vllm.sh outputs/model -d 0,1 -t 2 \
    --hccl-port 65000 -q
```

### 5.2 NPU 图捕获失败

**症状**：启动时报 CUDA graph / 算子编译错误。

**解决方案**：
```bash
# 方案一：使用即时执行模式
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 \
    --enforce-eager -q

# 方案二：仅对 decode 阶段使用图捕获
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' -q
```

### 5.3 显存/NPU 内存不足

**症状**：OOM（Out of Memory）错误。

**解决方案**：
- 降低 `--gpu-memory` 参数（默认 0.8）
- 减小 `--max-model-len`（默认 4096）
- 使用更多设备增加张量并行度

### 5.4 量化模型加载失败

**症状**：vLLM 无法识别量化模型格式。

**排查步骤**：
1. GPU 模型确认使用标准 GPTQ 或 INT8 格式
2. NPU 模型确认包含 `quant_model_description.json`
3. NPU 模型使用 `-q` 参数或确保 `--quantization ascend` 已指定
4. 如为 GPU 格式 GPTQ 模型需在 NPU 部署，先使用 `gptq_gpu_to_npu.py` 转换

### 5.5 服务端口冲突

**症状**：端口已被占用，服务无法启动。

**解决方案**：
```bash
# 使用其他端口
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 --port 9090
```

### 5.6 日志查看

默认日志保存在 `logs/vllm_server/` 目录下，文件名格式为 `{模型名}_{时间戳}.log`：

```bash
# 查看最新日志
ls -lt logs/vllm_server/ | head -5

# 实时跟踪日志
tail -f logs/vllm_server/model_20260527_*.log
```
