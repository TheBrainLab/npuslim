# SparseGPT 稀疏化算法

## 算法原理

SparseGPT 是一种训练后稀疏化（Post-Training Sparsification）算法，由 Frantar & Alistarh 在 2023 年提出。与量化不同，稀疏化通过将部分权重置零来减少模型的有效参数量。SparseGPT 利用 Hessian 信息指导剪枝决策，并通过误差补偿将剪枝误差传播到未剪枝的权重上，最小化对模型输出的影响。

### 两种稀疏模式

#### 非结构化稀疏（Unstructured Sparsity）

按固定比例（如 50%）将权重置零，不受位置约束。剪枝准则为 Hessian 加权的幅值最小化：

$$\text{mask}[i,j] = \begin{cases} 0 & \text{if } w_{i,j}^2 / h_{jj}^{-2} \leq \text{threshold} \\ 1 & \text{otherwise} \end{cases}$$

即优先剪除 "Hessian 敏感度低 + 绝对值小" 的权重。

#### 半结构化稀疏（N:M Sparsity）

在每连续 $M$ 个权重中，恰好保留 $N$ 个非零值。最常用的是 **2:4 稀疏**（每 4 个权重中保留 2 个），在 NVIDIA GPU（通过稀疏张量核心）和华为 Ascend NPU 上均有硬件加速支持。

2:4 稀疏的剪枝决策在滑动窗口内执行：

```python
for i in range(0, columns, prunem):
    window = min(prunem, count - i)
    k = min(prunen, window)
    # 在窗口内选择 Hessian 加权幅值最大的 k 个权重保留
    scores = W[:, i:i+window]**2 / diag(Hinv)[i:i+window]**2
    mask.scatter_(1, i + topk(scores, k, largest=False)[1], True)
```

### 误差补偿

与 GPTQ 类似，SparseGPT 在剪枝每个权重后，通过 Hessian 逆矩阵将误差传播到后续未剪枝的权重上：

$$w_{:,j} \leftarrow w_{:,j} - \frac{w_{:,j} - 0}{[H^{-1}]_{jj}} \cdot H^{-1}_{j,:}$$

这使得被剪枝权重"释放"的容量被重新分配到保留的权重上，保持层输出的总体精度。

## 实现细节

### 整体架构

| 组件 | 位置 | 职责 |
|------|------|------|
| `SparseGPTAlgorithm` | `algorithms/quantization/sparsegpt/sparsegpt_algo.py` | 顶层算法类 |
| `SparseGPTModule` | 同上 | 逐层稀疏化处理，包含 `fasterprune()` 核心逻辑 |
| `AscendSparse24Linear` | 同上 | NPU 2:4 稀疏权重打包与推理 |
| `_SparseMode` | 同上 | 平台 x 稀疏类型矩阵 |

### 稀疏模式矩阵

`_SparseMode` 枚举自动根据目标后端和稀疏参数选择处理模式：

```
                    prunen > 0       prunen == 0
                 (半结构化稀疏)    (非结构化稀疏)
NPU 后端    NPU_STRUCTURED    NPU_UNSTRUCTURED
GPU 后端    GPU_STRUCTURED    GPU_UNSTRUCTURED
```

不同模式决定了张量的保存格式：

| 模式 | 输出格式 | 说明 |
|------|---------|------|
| `GPU_UNSTRUCTURED` | 原始浮点权重（含零值） | 伪量化格式 |
| `GPU_STRUCTURED` | 原始浮点权重（含零值） | 伪量化格式 |
| `NPU_UNSTRUCTURED` | 原始浮点权重（含零值） | 伪量化格式 |
| `NPU_STRUCTURED` | int8 密集值 + tiled index | Sparse24 打包格式 |

### fasterprune 核心流程

```python
def fasterprune(self):
    W = self.layer.weight.data.float().clone()
    Hinv = self.compute_hinv(H)

    for i1 in range(0, columns, blocksize):
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        # 构建剪枝 mask
        if prunen == 0:
            # 非结构化：基于全局阈值
            tmp = W1**2 / diag(Hinv1)**2
            thresh = sort(tmp)[int(numel * sparsity)]
            mask = tmp <= thresh
        else:
            # N:M 半结构化：滑动窗口内选择
            for i in range(count):
                if i % prunem == 0:
                    scores = W[:, i:i+window]**2 / diag(Hinv)[i:i+window]**2
                    mask.scatter_(topk(scores, prunen, largest=False))

        # 逐列剪枝 + 误差补偿
        for i in range(count):
            q = w.clone()
            q[mask[:, i]] = 0              # 剪枝
            err = (w - q) / d              # 误差
            W1[:, i:] -= err @ Hinv[i, i:] # 误差传播

        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]  # 块间误差传播
```

### Ascend NPU 2:4 稀疏打包

当目标后端为 NPU 且使用 2:4 稀疏时，`AscendSparse24Linear` 将剪枝后的浮点权重打包为硬件友好的格式：

1. **Per-channel 对称量化**到 int8
2. **提取非零值**：每 4 个元素中保留 2 个非零值，形成 `[out, in//2]` 的密集矩阵
3. **生成索引**：编码每个非零值的原始位置，采用 tiled 布局（N 维度按 16 对齐，K 维度按 8 分块）

| 张量 | 形状 | 说明 |
|------|------|------|
| `weight` | `[out, in//2]` int8 | 密集化的非零值 |
| `weight_scale` | `[out]` | per-channel 量化缩放因子 |
| `weight_index` | 1D uint8 (tiled) | 非零值位置索引 |

推理时通过 AscendC `sparse_matmul_4to2` 算子执行稀疏矩阵乘法。

## 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sparsity` | float | 0.0 | 非结构化稀疏度（0.0~1.0），如 0.5 表示 50% 稀疏 |
| `prunen` | int | 0 | N:M 稀疏中的 N（保留数量），0 表示使用非结构化稀疏 |
| `prunem` | int | 0 | N:M 稀疏中的 M（窗口大小），0 表示使用非结构化稀疏 |
| `blocksize` | int | 128 | 逐列处理的块大小 |
| `percdamp` | float | 0.01 | Hessian 阻尼系数 |
| `preproc_hessian` | bool | True | 是否预处理 Hessian（处理 dead 行） |
| `fake_quant` | bool | False | 伪量化模式，保留浮点权重不打包 |
| `max_calib_samples` | int | 128 | 最大校准样本数 |
| `save_backend` | str | None | 强制指定保存后端 |

### 参数约束

- `sparsity` 和 `prunen:prunem` 互斥：同时设置时 N:M 模式优先，`sparsity` 被忽略并产生警告
- NPU 2:4 打包要求 `prunen=2, prunem=4`，其他 N:M 比例仅支持伪量化输出
- `prunen <= prunem`，且 `prunem > 0`（当 `prunen > 0` 时）
- `percdamp` 须在 `[0, 1)` 范围内

## 完整 YAML 配置示例

### 非结构化 50% 稀疏（GPU）

```yaml
metadata:
  name: "SparseGPT_Unstructured_Recipe"
  description: "SparseGPT unstructured 50% sparsity on GPU"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: cuda

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "SparseGPT_Pruning"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: SparseGPT
      sparsity: 0.5
      blocksize: 128
      percdamp: 0.01
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 2:4 结构化稀疏（NPU 打包）

```yaml
metadata:
  name: "SparseGPT_Sparse24_NPU_Recipe"
  description: "SparseGPT 2:4 structured pruning with Ascend sparse24 packing"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: npu

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096

recipe:
  - name: "SparseGPT_Sparse24_Pruning"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: SparseGPT
      prunen: 2
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

### 2:4 稀疏 + 跳过 down_proj

```yaml
metadata:
  name: "SparseGPT_Sparse24_Skip_Recipe"
  description: "2:4 sparsity with down_proj skipped"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: npu

  - id: calib_data
    type: TextDataset
    data_path: dataset/sharegpt_gpt4_qwen/sharegpt_gpt4-qwen3_a22B_output.jsonl
    num_samples: 128
    max_seq_length: 4096

recipe:
  - name: "SparseGPT_Sparse24_Pruning"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
    algorithm:
      type: SparseGPT
      prunen: 2
      prunem: 4
      fake_quant: false
    ignore_layers:
      - model.layers.*.mlp.down_proj
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 伪量化模式（调试用）

伪量化模式执行完整的剪枝流程，但输出原始浮点权重而非打包格式，适合验证稀疏化精度：

```yaml
algorithm:
  type: SparseGPT
  prunen: 2
  prunem: 4
  fake_quant: true        # 保留浮点权重，不执行 sparse24 打包
```

## ignore_layers 配置

```yaml
# 跳过所有 MLP 的 down_proj（该层对稀疏化更敏感）
ignore_layers:
  - "model.layers.*.mlp.down_proj"

# 跳过最后的 lm_head
ignore_layers:
  - "lm_head"

# 组合使用
ignore_layers:
  - "lm_head"
  - "model.layers.*.mlp.down_proj"
```

跳过的层保留原始浮点权重，不参与剪枝。实际经验表明，MLP 的 `down_proj` 层对稀疏化较敏感，跳过它可以显著提升稀疏模型精度。

## 量化元数据

NPU 2:4 稀疏模式下的 `ascend_quant_config`：

```python
{
    "model_quant_type": "Sparse24",
    "quant_layer_types": ["AscendSparse24Linear"],
    "sparsity_type": "2:4",
}
```

非 NPU 模式或伪量化模式下不写入量化元数据。

## 稀疏模式选择建议

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| NPU 推理加速 | 2:4 结构化（`prunen=2, prunem=4`） | 硬件原生支持，推理加速 2x |
| GPU 推理加速 | 2:4 结构化（需 Ampere+ 架构） | NVIDIA 稀疏张量核心支持 |
| 最大压缩比 | 非结构化 50%（`sparsity=0.5`） | 压缩比高但硬件加速有限 |
| 精度敏感场景 | 非结构化 30%（`sparsity=0.3`） | 较低稀疏度，精度损失更小 |

## 与量化算法的组合

SparseGPT 可以与量化算法组合使用（两阶段流水线）：

1. **先稀疏后量化**：使用 SparseGPT 剪枝，再用 GPTQ/INT8Dynamic 量化非零权重
2. **独立使用**：单独使用 SparseGPT 进行稀疏化，非零权重保持原始精度

当前 NPUSlim v2 中 SparseGPT 作为独立算法使用。如需组合量化，可在第一个 recipe 任务完成稀疏化后，在输出模型上运行第二个量化 recipe。
