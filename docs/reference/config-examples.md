# 配置文件样例集

本文档按算法分类，给出每个算法在不同模型上的 YAML 配置。所有配置均取自 `configs/` 目录下的实际文件。

---

## 配置结构概览

```yaml
metadata:
  name: "Recipe_Name"
  description: "..."

resources:                         # 资源声明（模型 + 数据集）
  - id: <id>                      # recipe 中通过 @id 引用
    type: <注册类型>               # Qwen3Model / OPTModel / GLM5
    path: <模型路径>
    model_hub: hf|ms              # 默认 hf
    device_map: cuda|cpu|npu
    trust_remote_code: true       # 部分模型需要
    low_cpu_mem_usage: true

  - id: <data_id>
    type: C4Dataset|TextDataset
    num_samples: 128
    max_seq_length: 2048
    # data_path: ...              # TextDataset 必填

recipe:
  - name: "Task_Name"
    type: compressor
    model: "@<模型id>"
    dataloader:                   # INT8Dynamic 无需此段
      dataset: "@<数据集id>"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: <算法类型>
    ignore_layers: []             # glob 模式，如 model.layers.*.mlp.down_proj
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

---

## 1. INT8 Dynamic

per-channel 权重 + per-token 激活量化。**不需要校准数据集**。

### Qwen3-8B (GPU)

```yaml
metadata:
  name: "Qwen3_Quantization_Recipe"
  description: "INT8 dynamic quantization for Qwen3-8B."

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
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### OPT-125M (GPU)

将 resources 替换为 `type: OPTModel`, `path: facebook/opt-125m`, `model: "@opt"`，其余不变。

### Qwen3-VL-30B-A3B (GPU, 扩展参数)

```yaml
resources:
  - id: qwen3_vl
    type: Qwen3VLModel
    path: Qwen/Qwen3-VL-30B-A3B-Thinking
    model_hub: ms
    device_map: cuda
    trust_remote_code: true
    low_cpu_mem_usage: true

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3_vl"
    algorithm:
      type: INT8Dynamic
      wbits: 8
      w_quant_method: per-channel    # 权重量化方式
      a_quant_method: per-token      # 激活量化方式
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

> **NPU 适配**：将 `device_map` 改为 `npu` 即可。输出自动生成 `quant_model_description.json`。

---

## 2. GPTQ

基于 Hessian 统计的激活感知权重量化。**需要校准数据集**。

核心 algorithm 参数：`wbits: 4`, `groupsize: 128`。

### Qwen3-0.6B W4A16 (CPU)

```yaml
metadata:
  name: "Qwen3_Quantization_Recipe"
  description: "GPTQ W4A16 quantization for Qwen3-0.6B."

resources:
  - id: qwen
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: cpu

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "GPTQ_Quantization"
    type: compressor
    model: "@qwen"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: GPTQ
      wbits: 4
      groupsize: 128
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### Qwen3-8B W4A16 (CPU)

同上，仅替换 `path: Qwen/Qwen3-8B`。

### Qwen3-235B-A22B W4A16 (CPU, ModelScope, MoE)

```yaml
resources:
  - id: qwen
    type: Qwen3Model
    path: Qwen/Qwen3-235B-A22B-Instruct-2507
    device_map: cpu
    model_hub: ms

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096       # MoE 大模型使用更长序列
```

> MoE 模型使用 `TextDataset`（本地 JSONL）替代 `C4Dataset`。

### Qwen3-30B-A3B W4A16 (CPU, ModelScope)

同 235B 结构，替换 `path: Qwen/Qwen3-30B-A3B-Instruct-2507`，`chunk_size: 8`。

### OPT-125M W4A16 (GPU)

```yaml
resources:
  - id: opt
    type: OPTModel
    path: facebook/opt-125m
    device_map: cuda            # GPU 模式
  # calib_data 同上（C4Dataset）
```

### GLM-5 W4A16 (CPU)

```yaml
resources:
  - id: glm5
    type: GLM5                  # 注意：非 GLM5Model
    path: /data16t/npu_bak/modelscope/GLM-5
    trust_remote_code: true     # 必需
    low_cpu_mem_usage: true
    device_map: cpu
  # calib_data: C4Dataset, 128 samples, seq 2048
```

---

## 3. QuIP

基于非相干处理的量化方法。**需要校准数据集 + GPU**。

algorithm 参数：

| 参数 | 说明 |
|------|------|
| `wbits` | 量化位数 |
| `quant_func` | 量化核函数，`rms` = 均方根 |
| `ldlq_method` | LDLQ 分解方法 |
| `npasses` | 迭代次数，0 = 单次 |
| `incoh_processing` | 非相干处理开关 |

### Qwen3-0.6B W4A16 (GPU)

```yaml
resources:
  - id: qwen
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: cuda            # QuIP 需要 GPU

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "QuIP_Quantization"
    type: compressor
    model: "@qwen"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: QuIP
      wbits: 4
      quant_func: rms
      ldlq_method: ldlq
      npasses: 0
      incoh_processing: true
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### OPT-125M W4A16 (GPU)

替换 resources：`type: OPTModel`, `path: facebook/opt-125m`，algorithm 不变。

---

## 4. SparseGPT

一次性训练后剪枝，支持非结构化和 2:4 结构化稀疏。**需要校准数据集**。

algorithm 参数：

| 参数 | 说明 |
|------|------|
| `sparsity` | 非结构化稀疏度（如 0.5 = 50%） |
| `prunen` / `prunem` | 2:4 结构化稀疏参数（设为 2/4） |
| `blocksize` | Hessian 计算块大小，默认 128 |
| `percdamp` | Hessian 阻尼百分比，默认 0.01 |
| `fake_quant` | `false` = 真实打包；`true` = 仅置零不打包 |

> 非结构化用 `sparsity`，结构化用 `prunen`+`prunem`，两者互斥。

### 4.1 非结构化稀疏

#### Qwen3-0.6B 50% 稀疏 (CPU)

```yaml
resources:
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: cpu

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096

recipe:
  - name: "SparseGPT_Pruning"
    type: compressor
    model: "@qwen3"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: SparseGPT
      sparsity: 0.5             # 50% 非结构化稀疏
      blocksize: 128
      percdamp: 0.01
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

#### OPT-125M 50% 稀疏 (GPU)

替换 resources：`type: OPTModel`, `path: facebook/opt-125m`, `device_map: cuda`，`C4Dataset`（seq 2048）。

### 4.2 2:4 结构化稀疏（Sparse24）

#### Qwen3-0.6B Sparse24 (CPU)

```yaml
    algorithm:
      type: SparseGPT
      prunen: 2                 # 每 4 个权重中剪枝 2 个
      prunem: 4                 # 2:4 结构化稀疏
      blocksize: 128
      percdamp: 0.01
      fake_quant: false         # 真实稀疏打包
```

#### Qwen3-0.6B Sparse24 + skip down_proj (CPU)

```yaml
    algorithm:
      type: SparseGPT
      prunen: 2
      prunem: 4
      blocksize: 128
      percdamp: 0.01
      fake_quant: false
    ignore_layers:
      - model.layers.*.mlp.down_proj
```

#### Qwen3-0.6B Sparse24 Fake Quant (CPU)

```yaml
    algorithm:
      type: SparseGPT
      prunen: 2
      prunem: 4
      blocksize: 128
      percdamp: 0.01
      fake_quant: true          # 仅标记掩码不打包
    ignore_layers:
      - model.layers.*.mlp.down_proj
```

> `fake_quant: true` 仅将剪枝位置置零，用于精度评估或不支持稀疏格式的后端。

#### OPT-125M Sparse24 (NPU)

```yaml
resources:
  - id: opt
    type: OPTModel
    path: Xorbits/opt-125m
    model_hub: ms
    device_map: npu             # NPU 模式

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096
```

> `device_map: npu` 时输出自动包含 `quant_model_description.json`。

---

## 速查表

### 算法参数

| 算法 | 类型 | 必需参数 | 可选参数 | 校准数据 |
|------|------|----------|----------|:--------:|
| INT8Dynamic | `INT8Dynamic` | `wbits` | `w_quant_method`, `a_quant_method` | 否 |
| GPTQ | `GPTQ` | `wbits`, `groupsize` | — | 是 |
| QuIP | `QuIP` | `wbits` | `quant_func`, `ldlq_method`, `npasses`, `incoh_processing` | 是 |
| SparseGPT | `SparseGPT` | — | `sparsity` 或 `prunen`+`prunem`, `blocksize`, `percdamp`, `fake_quant` | 是 |

### 数据集类型

| 类型 | 说明 | 参数 |
|------|------|------|
| `C4Dataset` | C4 在线数据集（流式加载+本地缓存） | `num_samples`, `max_seq_length` |
| `TextDataset` | 本地文本数据集（JSONL/Parquet） | `data_path`, `num_samples`, `max_seq_length` |

### 设备适用性

| 算法 | GPU | CPU | NPU |
|------|:---:|:---:|:---:|
| INT8Dynamic | 支持 | 支持 | 支持 |
| GPTQ | 支持 | 支持（推荐） | 支持 |
| QuIP | 支持（推荐） | — | — |
| SparseGPT | 支持 | 支持 | 支持 |
