# 配置体系

NPUSlim v2 采用 YAML 配置文件驱动整个量化/压缩流水线。配置文件由三大顶层块组成：`metadata`（元信息）、`resources`（资源声明）和 `recipe`（执行配方）。本文档详细说明配置文件的格式、字段语义以及解析校验流程。

---

## YAML 结构概览

```yaml
metadata:
  name: "..."
  description: "..."

resources:
  - id: ...
    type: ...
    # 资源特有字段 ...

recipe:
  - name: "..."
    type: compressor
    model: "@resource_id"
    dataloader: { ... }
    algorithm: { ... }
    ignore_layers: [ ... ]
    execution: { ... }
    saver: { ... }
```

---

## metadata 字段

`metadata` 块用于描述当前配置的元信息，字段如下：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 否 | 配方名称，用于日志标识和输出路径推导 |
| `description` | string | 否 | 配方描述 |

`metadata` 还可包含非标准字段（如 `log_dir`、`work_dir`），这些字段在 `resolve_log_dir()` 中用于覆盖默认日志路径。

---

## resources 字段详解

`resources` 是一个列表，声明流水线所需的所有外部资源（模型、数据集等）。每个资源项包含通用字段和类型特有字段。

### 通用字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 资源唯一标识符，用于 recipe 中的 `@ref` 引用 |
| `type` | string | 是 | 资源类型名，对应 Registry 中注册的类名（不区分大小写） |

除 `id` 和 `type` 外，其余字段全部进入 `ResourceConfig.extra` 字典，在实例化时作为关键字参数传入。

### 模型资源特有字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | -- | HuggingFace 模型仓库 ID 或本地路径 |
| `model_hub` | string | `hf` | 模型仓库来源：`hf`（HuggingFace）或 `ms`（ModelScope） |
| `device_map` | string | `cpu` | 设备映射：`cpu`、`cuda`、`npu` |
| `trust_remote_code` | bool | `false` | 是否信任远程代码 |
| `low_cpu_mem_usage` | bool | `false` | 是否启用低 CPU 内存模式 |

### 数据集资源特有字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_samples` | int | `256` | 校准样本数量 |
| `max_seq_length` | int | `2048` | 最大序列长度 |
| `data_path` | string | -- | 本地数据文件路径（TextDataset 专用） |
| `seed` | int | `0` | 随机种子（C4Dataset 专用） |

### 资源声明示例

```yaml
resources:
  # 模型资源 — 从 HuggingFace 加载，部署到 CUDA
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    model_hub: hf
    device_map: cuda

  # 模型资源 — 从 ModelScope 加载，启用低内存模式
  - id: qwen3_vl
    type: Qwen3VLModel
    path: Qwen/Qwen3-VL-30B-A3B-Thinking
    model_hub: ms
    device_map: cuda
    trust_remote_code: true
    low_cpu_mem_usage: true

  # C4 数据集 — 从 HuggingFace Hub 流式加载
  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

  # Text 数据集 — 从本地 JSONL 文件加载
  - id: local_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096
```

---

## recipe 字段详解

`recipe` 是一个任务列表，定义了流水线的执行步骤。每个任务项对应一个 `RecipeTaskConfig` 实例。

### 通用字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 任务名称，用于日志标识 |
| `type` | string | 是 | 任务类型，目前支持 `compressor` |

### model — 模型引用

使用 `@id` 语法引用 resources 中声明的模型资源：

```yaml
model: "@qwen3"
```

`@` 前缀表示这是一个资源引用，解析器会自动去除前缀后在 resource 列表中查找匹配项。

### dataloader — 数据加载配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset` | string | 视算法而定 | 数据集资源引用，格式为 `@resource_id` |
| `batch_size` | int | 否 | 批大小，默认 1 |
| `shuffle` | bool | 否 | 是否打乱数据 |
| `pin_memory` | bool | 否 | 是否锁页内存（CUDA 场景推荐开启） |

> **注意**：INT8Dynamic 算法不需要校准数据，因此 `dataloader` 可省略。GPTQ、SparseGPT 等算法必须提供 `dataloader`。

### algorithm — 算法配置

`algorithm` 字典中 `type` 为必填字段，对应 `AlgorithmRegistry` 中注册的算法名。其余字段为算法特有参数：

**INT8Dynamic 算法参数：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | -- | 固定为 `INT8Dynamic` |
| `wbits` | int | `8` | 权重量化位宽 |
| `w_quant_method` | string | `per-channel` | 权重量化方法 |
| `a_quant_method` | string | `per-token` | 激活量化方法 |

**GPTQ 算法参数：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | -- | 固定为 `GPTQ` |
| `wbits` | int | `4` | 权重量化位宽 |
| `groupsize` | int | `128` | 分组大小 |

**QuIP 算法参数：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | -- | 固定为 `QuIP` |
| `wbits` | int | `4` | 权重量化位宽 |
| `quant_func` | string | `rms` | 量化函数类型 |
| `ldlq_method` | string | `ldlq` | LDLQ 方法 |
| `npasses` | int | `0` | 优化遍数 |
| `incoh_processing` | bool | `true` | 是否启用非相干处理 |

**SparseGPT 算法参数：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | -- | 固定为 `SparseGPT` |
| `prunen` | int | `2` | 每 m 列中保留 n 列（结构化剪枝） |
| `prunem` | int | `4` | 剪枝分组大小 |
| `blocksize` | int | `128` | 块大小 |
| `percdamp` | float | `0.01` | 阻尼百分比 |
| `fake_quant` | bool | `false` | 是否使用伪量化 |

### ignore_layers — 跳过层模式

支持 glob 和正则表达式模式，匹配的层将被跳过（不参与量化）。例如：

```yaml
ignore_layers:
  - "lm_head"            # 精确匹配
  - "model.layers.0.*"   # glob 模式
  - ".*bias$"            # 正则表达式
```

### execution — 执行配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | `streaming` | 执行模式，目前支持 `streaming` |
| `chunk_size` | int | `4` | 每个 chunk 包含的层数 |

`streaming` 模式下，模型按 chunk 分块加载、处理、保存，避免将完整模型驻留内存。

### saver — 保存配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 保存器类型，目前支持 `StreamingHuggingFaceSaver` |
| `save_dir` | string | 是 | 输出根目录 |
| `save_path` | string | 否 | 精确输出路径（优先级高于 save_dir 的推导） |

当使用 `save_dir` 时，实际输出路径为 `save_dir/<config相对路径>`。例如配置文件路径为 `configs/qwen3/gptq/qwen3_8b-w4a16.yaml`，则输出到 `./outputs/qwen3/gptq/qwen3_8b-w4a16/`。

---

## 解析与校验流程

### 解析流程

入口函数 `bootstrap_from_path()` 完成从 YAML 文件到可用 `EngineConfig` 的全过程：

```
YAML 文件
  |
  v
_load_raw_yaml()          # yaml.safe_load 得到原始 dict
  |
  v
parse_config()            # dict -> EngineConfig
  |  +-- MetadataConfig    # metadata 块 -> dataclass
  |  +-- ResourceConfig[]  # resources 列表，id/type 以外进入 extra
  |  +-- RecipeTaskConfig[]# recipe 列表，标准字段提取后其余进入 extra
  |
  v
apply_saver_path_policy() # 从配置文件路径推导 saver 输出路径
  |
  v
validate_config()         # 校验 @ref 引用合法性
  |
  v
print_config()            # Rich 表格打印配置摘要
```

**关键设计**：`ResourceConfig` 和 `RecipeTaskConfig` 均包含 `extra: Dict[str, Any]` 字段，用于捕获所有非标准键。这使得配置格式可扩展，无需修改 schema 即可支持新参数。

### 校验流程

`validate_config()` 检查以下内容：

1. **资源 ID 唯一性**：resources 列表中不允许重复的 `id`。
2. **模型引用合法性**：recipe 中的 `model`、`main_model`、`draft_model` 字段若以 `@` 开头，必须指向 resources 中已声明的资源。
3. **数据集引用合法性**：`dataloader.dataset` 必须以 `@` 引用已声明的资源。
4. **废弃字段检测**：若任务中存在 `data` 字段（旧语法），将报错提示迁移至 `dataloader.dataset`。
5. **空列表警告**：resources 或 recipe 为空时发出警告。

校验失败时抛出 `ValidationError` 异常。开启 `strict=True` 模式时，警告也会升级为错误。

---

## 完整配置示例

### INT8 动态量化（无校准数据）

```yaml
metadata:
  name: "Qwen3_INT8_Recipe"
  description: "INT8 dynamic quantization for Qwen3-8B on CUDA."

resources:
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    model_hub: hf
    device_map: cuda

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"
    algorithm:
      type: INT8Dynamic
      wbits: 8
      w_quant_method: per-channel   # 权重逐通道量化
      a_quant_method: per-token     # 激活逐 token 量化
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### GPTQ 量化（带 C4 校准数据）

```yaml
metadata:
  name: "OPT_GPTQ_Recipe"
  description: "GPTQ W4A16 quantization for OPT-125M."

resources:
  - id: opt
    type: OPTModel
    path: facebook/opt-125m
    device_map: cuda

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "GPTQ_Quantization"
    type: compressor
    model: "@opt"
    dataloader:
      dataset: "@calib_data"    # 引用上面声明的校准数据集
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: GPTQ
      wbits: 4                  # 4-bit 权重量化
      groupsize: 128            # 128 为一组进行分组量化
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 使用本地 Text 数据集的 SparseGPT

```yaml
metadata:
  name: "Qwen3_SparseGPT_Recipe"
  description: "SparseGPT 2:4 structured pruning for Qwen3-0.6B."

resources:
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    model_hub: hf
    device_map: cpu

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096

recipe:
  - name: "SparseGPT_Sparse24_Pruning"
    type: compressor
    model: "@qwen3"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: SparseGPT
      prunen: 2                  # 2:4 结构化稀疏
      prunem: 4
      blocksize: 128
      percdamp: 0.01
      fake_quant: false
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

---

## 运行配置

配置文件编写完成后，通过 CLI 工具执行：

```bash
# GPU 上的 INT8 量化
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml

# GPU 上的 GPTQ 量化
python tools/run.py -c configs/opt/gptq/opt_125m-w4a16.yaml

# NPU 上的量化（在配置文件中将 device_map 设为 npu）
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

如遇 HuggingFace 下载受限，可设置镜像环境变量：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```
