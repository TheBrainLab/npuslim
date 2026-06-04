# CLI 工具参考文档

本文档涵盖 NPUSlim v2 框架提供的三个命令行工具：量化主程序 `run.py`、交互式聊天客户端 `chat.py`，以及 GPU->NPU 格式转换工具 `gptq_gpu_to_npu.py`。

---

## 1. run.py — 量化主程序

### 用法

```bash
python tools/run.py -c <config.yaml>
python tools/run.py <config.yaml>
```

`config` 参数支持位置参数和 `-c`/`--config` 两种形式，二选一即可。

### 命令行参数

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `config` | — | 位置参数 | 无 | 配置文件路径（YAML），与 `-c` 二选一 |
| `-c` / `--config` | `-c` | str | 无 | 配置文件路径（YAML），与位置参数二选一 |
| `--log-dir` | — | str | `None` | 日志与配置快照的输出目录；不指定时使用框架默认行为 |
| `--no-header` | — | flag | `False` | 禁止启动时打印 NPUSlim ASCII 横幅 |

### 环境变量

| 变量名 | 用途 |
|--------|------|
| `HF_ENDPOINT` | HuggingFace 镜像地址。当无法访问 HuggingFace 时设为 `https://hf-mirror.com` |
| `ASCEND_HOME_PATH` | CANN 安装路径。使用 NPU 量化前必须设置，指向 CANN 根目录 |

### 典型用法示例

**GPU INT8 动态量化（Qwen3-8B）**

```bash
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

**GPU GPTQ 量化（OPT-125M）**

```bash
python tools/run.py -c configs/opt/gptq/opt_125m-w4a16.yaml
```

**NPU GPTQ 量化**

在配置文件中将 `device_map` 设为 `npu`，然后：

```bash
python tools/run.py -c configs/opt/sparsegpt/opt_125m-sparse24.yaml
```

**使用 HuggingFace 镜像**

```bash
export HF_ENDPOINT="https://hf-mirror.com"
python tools/run.py -c configs/qwen3/gptq/qwen3_8b-w4a16.yaml
```

**自定义日志目录**

```bash
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml --log-dir ./my_logs
```

### 输出目录结构

量化完成后，输出目录（由配置中 `saver.save_dir` 指定，默认 `./outputs`）包含：

```
outputs/
├── model-00001-of-0000X.safetensors   # 量化后的模型分片
├── model.safetensors.index.json       # 分片索引
├── config.json                        # 模型配置（含/不含 quantization_config）
├── tokenizer.json                     # 分词器文件
├── tokenizer_config.json              # 分词器配置
├── special_tokens_map.json            # 特殊 token 映射
├── generation_config.json             # 生成配置
├── quant_model_description.json       # NPU 模式下生成的量化描述文件
└── ...                                # 其他模型附带的文件
```

### 执行流程

1. 解析命令行参数，定位配置文件
2. `bootstrap_from_path()` 加载 YAML，解析为 `EngineConfig`，校验资源引用
3. `SlimEngine` 根据配置创建 `ResourceManager`，构建任务列表
4. 依次执行 recipe 中的每个任务（`compressor` 类型执行流式量化）
5. 任务通过 `ChunkLoader` 逐块加载权重、调用算法、通过 `StreamingHuggingFaceSaver` 逐块写出

---

## 2. chat.py — 交互式聊天客户端

基于 OpenAI Python SDK 的 vLLM 兼容 API 交互式聊天工具。支持流式输出和富文本显示（需安装 `rich`）。

### 用法

```bash
python tools/chat.py [选项]
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--url` | str | `http://127.0.0.1:8090/v1` | vLLM OpenAI 兼容 API 地址 |
| `--model` | str | （内置路径） | 模型名称或路径；不指定时自动从 API 获取 |
| `--max-tokens` | int | `2048` | 单次生成的最大 token 数 |
| `--temperature` | float | `0.0` | 采样温度（0.0 = 贪心） |
| `--top-p` | float | `1.0` | Top-p 核采样阈值 |
| `--chat` | flag | `False` | 使用 Chat Completions API（应用模型 chat template）；否则使用 Completions API |
| `--no-history` | flag | `False` | 禁用多轮对话历史（每轮独立） |

### 交互命令

聊天过程中支持以下命令：

| 命令 | 说明 |
|------|------|
| `/clear` | 清屏并清除对话历史 |
| `/exit`、`/quit`、`/q` | 退出聊天 |
| `/help` | 显示帮助信息 |
| `/model` | 显示当前模型名称 |
| `/temp <值>` | 动态调整采样温度 |
| `/topp <值>` | 动态调整 top-p |

### 使用示例

**基本 Completion 模式**

```bash
# 先启动 vLLM 服务
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# 启动聊天
python tools/chat.py --url http://127.0.0.1:8090/v1
```

**Chat 模式（多轮对话）**

```bash
python tools/chat.py --url http://127.0.0.1:8090/v1 --chat
```

**单轮无历史**

```bash
python tools/chat.py --url http://127.0.0.1:8090/v1 --chat --no-history
```

**指定模型和采样参数**

```bash
python tools/chat.py \
    --url http://127.0.0.1:8090/v1 \
    --model Qwen/Qwen3-8B \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 4096
```

### 依赖

- `openai` — 必需
- `rich` — 可选，提供彩色输出和 Markdown 渲染

---

## 3. gptq_gpu_to_npu.py — GPU->NPU 格式转换

将标准 GPTQ 量化模型（GPU 格式）转换为 vLLM-Ascend 所需的 NPU (Ascend) 格式。

### 用法

```bash
python tools/convert/gptq_gpu_to_npu.py -i <输入模型路径> -o <输出模型路径>
```

### 命令行参数

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | str | **必需** | GPU 格式 GPTQ 模型目录路径 |
| `--output` | `-o` | str | **必需** | 输出 NPU 格式模型目录路径 |
| `--device` | — | str | `cpu` | 转换使用的设备（推荐 `cpu`，避免大模型 GPU OOM） |
| `--verbose` | `-v` | flag | `False` | 启用详细日志（DEBUG 级别） |

### 转换逻辑

转换过程将 GPU 格式的 GPTQ 权重重打包为 Ascend 列式打包格式：

| 项目 | GPU 格式（输入） | NPU 格式（输出） |
|------|------------------|-------------------|
| 权重张量 | `qweight` [infeatures//8, outfeatures] int32 | `weight` [outfeatures, infeatures//8] int32 |
| 缩放因子 | `scales` [num_groups, outfeatures] float16 | `weight_scale` [outfeatures, num_groups] bfloat16 |
| 零点 | `qzeros` [num_groups, outfeatures//8] int32 | `weight_offset` [outfeatures, num_groups] bfloat16 |
| 量化配置 | `config.json` 中的 `quantization_config` | 移除 `quantization_config`，新增 `ascend_quant_config` |
| 量化描述 | 无 | 新增 `quant_model_description.json` |

关键处理步骤：

1. 从 `config.json` 的 `quantization_config` 读取量化参数（bits, group_size）
2. 解包 GPU int32 打包权重为独立 4-bit 值
3. 将无符号 [0,15] 转为有符号 [-8,7] 表示
4. 按 Ascend 列式格式重打包（8 个 int4 打包进一个 int32）
5. 转换 scales/zeros 为 weight_scale/weight_offset 格式
6. 移除原 `quantization_config`，写入 `ascend_quant_config` 和 `quant_model_description.json`

### 使用示例

**基本转换**

```bash
python tools/convert/gptq_gpu_to_npu.py \
    -i /path/to/gpu_gptq_model \
    -o /path/to/npu_model
```

**详细日志**

```bash
python tools/convert/gptq_gpu_to_npu.py \
    -i /path/to/gpu_gptq_model \
    -o /path/to/npu_model \
    -v
```

### 限制

- 仅支持 4-bit 量化模型（bits=4）
- 转换过程默认在 CPU 上执行（大模型推荐保持默认，避免 GPU 内存不足）
- 输入模型必须包含 `config.json` 中的 `quantization_config` 字段
- 输出模型的 float16 权重会自动转为 bfloat16（NPU 要求）

### 依赖

- PyTorch
- Transformers
- NPUSlim（`GPTQQuantLinear` 类）
- loguru, tqdm, safetensors
