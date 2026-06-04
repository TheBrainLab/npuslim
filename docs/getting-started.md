# 快速上手

本文档帮助你从零开始安装 NPUSlim 并完成第一次模型量化。

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | >= 3.11 | 框架使用现代类型语法 |
| PyTorch | >= 2.0 | GPU 或 Ascend 扩展均可 |
| Git | 任意 | 克隆仓库 |
| 华为 CANN | 8.0+ | **仅 NPU 后端需要** |

## 安装

### 基础安装（GPU）

```bash
git clone <repo-url> && cd npuslim
pip install -e .
```

### NPU / Ascend 安装

在安装前确保 CANN 工具包已部署并设置环境变量：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
pip install -e ".[npu]"
```

### 可选依赖

```bash
pip install -e ".[eval]"     # 评估工具 (lm-eval, evalscope)
pip install -e ".[vllm]"     # vLLM 部署支持
pip install -e ".[all]"      # 安装全部可选依赖
```

## 第一个量化任务

以下示例使用 INT8 动态量化对 Qwen3-8B 进行压缩。整个过程采用流式管线，峰值显存远低于全量加载。

### 1. 准备配置文件

框架内置了样例配置，位于 `configs/` 目录。也可直接编写 YAML：

```yaml
metadata:
  name: "Qwen3_INT8_Recipe"
  description: "INT8 dynamic quantization for Qwen3-8B"

resources:
  - id: qwen3
    type: Qwen3
    path: Qwen/Qwen3-8B
    model_hub: hf
    device_map: cuda

  - id: calib_data
    type: C4
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
    algorithm:
      type: INT8Dynamic
      wbits: 8
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 2. 运行量化

```bash
# 如无法访问 HuggingFace，可使用镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 执行量化
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

### 3. GPU vs NPU

两者仅在配置文件的 `device_map` 字段不同：

| 后端 | device_map 值 | 额外要求 |
|------|--------------|---------|
| GPU (CUDA) | `cuda` | 无 |
| CPU | `cpu` | 无（速度较慢） |
| NPU (Ascend) | `npu` | 安装 CANN 并设置 `ASCEND_HOME_PATH` |

切换 NPU 只需修改配置：

```yaml
resources:
  - id: qwen3
    type: Qwen3
    path: Qwen/Qwen3-8B
    device_map: npu    # 改为 npu
```

NPU 模式下 `StreamingHuggingFaceSaver` 会自动生成 `quant_model_description.json`，供昇腾推理运行时使用。

## 输出目录结构

量化完成后，输出目录（默认 `./outputs/`）包含以下内容：

```
outputs/model/
├── config.json                         # 模型配置（含量化参数）
├── tokenizer.json                      # 分词器文件
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors.index.json        # 分片索引
├── model-00001-of-00004.safetensors    # 量化后权重分片
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── quant_model_description.json        # (NPU 模式) 昇腾推理描述文件
```

输出为标准 HuggingFace 格式，可直接用 `transformers` 加载，也可通过 vLLM 部署。

## 部署与验证

```bash
# 使用 vLLM 部署量化模型
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# 运行评估
bash tools/eval/run_lmeval.sh outputs/model --backend vllm --tasks wikitext -d 0
```

## 其他算法快速参考

| 算法 | 配置示例 | 说明 |
|------|---------|------|
| GPTQ W4A16 | `configs/qwen3/gptq/qwen3_0_6b-w4a16.yaml` | 基于 Hessian 统计的激活感知权重量化 |
| SparseGPT | `configs/qwen3/sparsegpt/qwen3_0_6b-sparse24.yaml` | 2:4 结构化稀疏 |
| QuIP | `configs/qwen3/quip/qwen3_0_6b-w4a16.yaml` | 非均匀量化 |

## 下一步

- **理解架构** — 阅读 [流式管线架构](design/streaming-pipeline.md) 了解核心设计思想
- **自定义配置** — 参考 [配置体系](guide/configuration.md) 学习完整配置字段
- **深入算法** — 查看 [量化算法总览](guide/quantization/overview.md) 选择适合的算法
- **深入内部** — 查看 [完整执行流程](internals/execution-flow.md) 了解全链路实现
- **部署评估** — 参考 [模型部署](deployment/serving.md) 进行 vLLM 部署与基准测试
- **插件开发** — 参考 [插件架构](plugins/plugin-architecture.md) 为 vLLM 或 Transformers 编写扩展
