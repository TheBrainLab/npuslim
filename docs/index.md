# NPUSlim v2 文档

## 项目简介

NPUSlim 是一个面向大语言模型（LLM）的**流式量化压缩框架**，旨在以极低的内存开销完成模型量化。框架采用 chunk-by-chunk 流式管线设计，无需将完整模型加载到内存即可完成量化处理，原生支持 GPU（CUDA）与华为昇腾 NPU（Ascend）双后端。

核心价值：让百亿参数模型的量化在单卡上即可完成，同时产出可直接部署的 HuggingFace 格式权重。

## 核心特性

- **流式量化管线** — 基于 `ChunkLoader` 分块加载、分块处理、增量写入，峰值显存仅与 chunk_size 相关
- **多算法支持** — 内置 INT8 动态量化（per-channel/per-token）、GPTQ、SparseGPT、QuIP 等
- **GPU + NPU 双后端** — 统一的 `BackendHandler` 抽象，配置中切换 `device_map` 即可
- **注册表驱动** — 算法、模型、数据集、任务、保存器均通过 `Registry` 懒加载注册，扩展便捷
- **声明式 YAML 配置** — 单一配置文件描述资源与流水线，支持 `@id` 引用复用资源
- **插件生态** — 通过 entry points 无缝集成 vLLM、vLLM-Ascend、HuggingFace Transformers
- **完整评估链** — 内置 LM-Eval Harness 评估与压力测试脚本，支持 vLLM / HF / API 三种推理后端

## 技术栈

| 组件 | 要求 |
|------|------|
| Python | >= 3.11 |
| PyTorch | 稳定版（GPU 或 Ascend 扩展） |
| 华为 CANN | NPU 后端需要，需设置 `ASCEND_HOME_PATH` |
| vLLM | 部署与评估（可选） |

## 文档导航

### 快速开始

| 文档 | 说明 |
|------|------|
| [快速上手](getting-started.md) | 环境安装、第一个量化任务、输出结构说明 |

### 设计思想 (design/)

| 文档 | 说明 |
|------|------|
| [流式管线架构](design/streaming-pipeline.md) | 流式分块管线的设计动机、数据流、内存管理策略 |
| [可扩展性设计](design/extensibility.md) | Registry 注册表模式、@resource 引用、扩展点总结 |

### 使用指南 (guide/)

| 文档 | 说明 |
|------|------|
| [配置体系](guide/configuration.md) | YAML schema 完整字段说明、解析与校验流程 |
| [校准数据](guide/calibration.md) | C4 / Text 数据集加载、自定义数据集扩展 |
| [量化算法总览](guide/quantization/overview.md) | 算法分类、对比表、选择指南 |
| [GPTQ 算法](guide/quantization/gptq.md) | 基于 Hessian 的激活感知权重量化 |
| [INT8 动态量化](guide/quantization/int8-dynamic.md) | per-channel/per-token 动态量化策略 |
| [QuIP 算法](guide/quantization/quip.md) | 不相干性处理量化方法 |
| [SparseGPT 算法](guide/quantization/sparsegpt.md) | Post-training 稀疏化方法 |
| [模型适配](guide/model-support.md) | 已支持模型详解与新模型接入指南 |

### 内部机制 (internals/)

| 文档 | 说明 |
|------|------|
| [完整执行流程](internals/execution-flow.md) | bootstrap → Engine → Task → Saver 全链路追踪 |
| [后端抽象](internals/backend.md) | CPU/CUDA/NPU 三端检测与 tensor 迁移 |
| [流式保存](internals/streaming-saver.md) | 分片写入 safetensors 与 Ascend 描述文件生成 |
| [自定义算子](internals/custom-ops.md) | sparse_matmul 等算子的构建与调用 |
| [Hook 系统](internals/hooks.md) | 生命周期钩子机制 |
| [分布式执行](internals/distributed.md) | 多卡/多节点量化支持 |

### 部署与评估 (deployment/)

| 文档 | 说明 |
|------|------|
| [模型部署](deployment/serving.md) | vLLM GPU/NPU 部署脚本详解 |
| [模型评估](deployment/evaluation.md) | LM-Eval 三种后端、压力测试、指标解读 |

### 插件生态 (plugins/)

| 文档 | 说明 |
|------|------|
| [插件架构](plugins/plugin-architecture.md) | patch 机制、register_patch 装饰器、版本兼容策略 |
| [vLLM 集成](plugins/vllm-integration.md) | MoE 模型补丁、推理执行器扩展 |
| [Ascend NPU 集成](plugins/ascend-integration.md) | W4A16 量化方案、method_adapter 补丁 |
| [Transformers 集成](plugins/transformers-integration.md) | 自定义量化器注册 |

### 参考资料 (reference/)

| 文档 | 说明 |
|------|------|
| [CLI 工具参考](reference/cli-reference.md) | run.py / chat.py 全部参数说明 |
| [配置文件样例](reference/config-examples.md) | 按算法/模型/后端分类的真实 YAML 示例 |

## 快速命令参考

```bash
# GPU 量化
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml

# NPU 量化（配置中 device_map 设为 npu）
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml

# 部署
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# 评估
bash tools/eval/run_lmeval.sh outputs/model --backend vllm --tasks wikitext -d 0
```
