# GPTQ 量化算法

## 算法原理

GPTQ（GPT Quantization）是一种基于二阶信息（Hessian 矩阵）的训练后权重量化算法，由 Frantar 等人在 2022 年提出。其核心思想是：**逐列量化权重，并通过 Hessian 逆矩阵将量化误差传播到未量化的列，以最小化对层输出的整体影响。**

### 数学描述

对于权重矩阵 $W \in \mathbb{R}^{r \times c}$，GPTQ 的目标是最小化：

$$\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

其中 $X$ 是校准数据，等价于最小化 $\text{tr}((W - \hat{W}) H (W - \hat{W})^T)$，$H = 2XX^T$ 为 Fisher 信息矩阵的近似。

GPTQ 采用 **OBQ（Optimal Brain Quantization）** 的贪心近似：按列依次量化，每量化一列后，利用 Hessian 逆将对角误差传播到后续列：

$$w_{:,j} \leftarrow w_{:,j} - \frac{w_{:,j} - \text{quant}(w_{:,j})}{[H^{-1}]_{jj}} \cdot H^{-1}_{j,:}$$

### 关键优化

- **Cholesky 分解**：预先计算 $H^{-1}$ 的上三角 Cholesky 分解，避免重复计算
- **Block 处理**：将列按 `blocksize` 分块，块内逐列量化并累积误差，块间一次性传播误差，减少内存访问
- **Activation Ordering**：按 Hessian 对角线降序排列列，优先量化"不重要"的列

## 实现细节

### 整体架构

GPTQ 实现由以下组件构成：

| 组件 | 位置 | 职责 |
|------|------|------|
| `GPTQAlgorithm` | `algorithms/quantization/gptq/gptq_algo.py` | 顶层算法类，管理 chunk 生命周期 |
| `GPTQModule` | 同上 | 逐层 GPTQ 优化，包含 `fasterquant()` 核心逻辑 |
| `GPTQQuantLinear` | 同上 | 量化权重打包，生成部署格式的张量 |
| `BaseHessianAlgorithm` | `algorithms/quantization/hessian/` | Hessian 累积、运行时模型管理等公共逻辑 |

### fasterquant 核心流程

```python
def fasterquant(self):
    # 1. 准备权重 (处理 Conv2d / Conv1D 的形状差异)
    w = self.layer.weight.data.float().clone()

    # 2. 计算 per-group scale 和 zero
    for i in range(0, columns, groupsize):
        scale, zero = compute_scales_with_zero(w[:, i:i+groupsize])

    # 3. Activation ordering: 按 Hessian 对角线降序排列
    perm = torch.argsort(torch.diag(H), descending=True)
    w, H = w[:, perm], H[perm][:, perm]

    # 4. Cholesky 分解 Hessian 逆
    hinv = self.compute_hinv(H)

    # 5. Block-wise 逐列量化 + 误差补偿
    for i1 in range(0, columns, blocksize):
        for i in range(count):
            q_col = quantize(w_col, scale, zero)       # 量化当前列
            err = (w_col - q_col) / d                   # 计算误差
            w1[:, i:] -= err @ hinv[i, i:]              # 块内误差传播
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]            # 块间误差传播

    # 6. 反转排列，恢复原始列顺序
    q = q[:, invperm]
```

### 权重打包格式

量化完成后，权重被打包为紧凑的整数格式：

**GPU (CUDA) 格式：**
- `qweight`：按 bit 打包的整数权重（如 4-bit 模式下每 32 列打包为一个 int32）
- `qzeros`：打包的零点
- `scales`：FP16 缩放因子
- `g_idx`：列到组的映射索引

**NPU (Ascend) 格式：**
- `weight`：按 8 列打包的 int32 权重（仅支持 4-bit）
- `weight_scale`：BF16 缩放因子
- `weight_offset`：BF16 零点偏移

## 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wbits` | int | 4 | 量化比特数，支持 2/3/4/8 |
| `groupsize` | int | 128 | 分组大小。-1 表示不分组（整个矩阵一组） |
| `actorder` | bool | True | 是否启用 activation ordering（按 Hessian 对角线排序） |
| `sym` | bool | True | 是否使用对称量化 |
| `blocksize` | int | 128 | 逐列量化的块大小 |
| `static_groups` | bool | True | 是否预先计算所有 group 的 scale/zero |
| `percdamp` | float | 0.01 | Hessian 阻尼系数，防止数值不稳定 |
| `preproc_hessian` | bool | True | 是否预处理 Hessian（处理 dead 行） |
| `fake_quant` | bool | False | 伪量化模式，不打包为整数，保留浮点权重 |
| `max_calib_samples` | int | 128 | 最大校准样本数 |
| `save_backend` | str | None | 强制指定保存后端（`npu`/`cuda`），覆盖自动检测 |

### 参数选择建议

- **`wbits`**：4-bit 是最常用的设置，在压缩率和精度之间取得最佳平衡；8-bit 几乎无损但压缩率较低
- **`groupsize`**：128 是通用推荐值。更小的值（如 32）精度更高但增加存储开销；-1（不分组）精度最低
- **`actorder`**：建议保持 True，能提升量化精度，特别是在低比特（2/3-bit）场景下效果显著
- **`sym`**：建议保持 True（对称量化），大多数推理框架的 GPTQ 实现基于对称量化

## 完整 YAML 配置示例

### GPU GPTQ W4A16

```yaml
metadata:
  name: "GPTQ_W4A16_Recipe"
  description: "GPTQ W4A16 quantization for Qwen3-8B on GPU"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    device_map: cuda

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "GPTQ_Quantization"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
      shuffle: true
      pin_memory: true
    algorithm:
      type: GPTQ
      wbits: 4
      groupsize: 128
      actorder: true
      sym: true
      percdamp: 0.01
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### NPU GPTQ W4A16

```yaml
metadata:
  name: "GPTQ_W4A16_NPU_Recipe"
  description: "GPTQ W4A16 quantization for Ascend NPU"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    device_map: npu

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "GPTQ_Quantization"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
    algorithm:
      type: GPTQ
      wbits: 4
      groupsize: 128
      save_backend: npu       # 强制使用 NPU 打包格式
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

## 支持的模型

| 模型 | 注册名 | 说明 |
|------|--------|------|
| Qwen3 系列 | `Qwen3Model` / `Qwen3` | 支持所有 Qwen3 规模 |
| OPT 系列 | `OPTModel` / `OPT` | Meta OPT 模型 |
| GLM-5 | `GLM5` | GlmMoeDsa 架构 |

## ignore_layers 配置

通过 `ignore_layers` 可以指定跳过量化的层，支持三种匹配模式：

```yaml
# 跳过特定层
ignore_layers:
  - "lm_head"

# 跳过所有 down_proj（glob 匹配）
ignore_layers:
  - "model.layers.*.mlp.down_proj"

# 使用正则表达式
ignore_layers:
  - "re:model\\.layers\\.[0-9]+\\.self_attn\\..*"
```

跳过的层将以原始 FP16/BF16 精度保存。

## 量化元数据

量化完成后，`GPTQAlgorithm` 会自动更新模型配置中的量化元数据：

**GPU 格式** (`quantization_config`)：
```python
{
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "desc_act": True,          # actorder
    "static_groups": True,
    "quant_method": "gptq",
    "checkpoint_format": "gptq",
    "true_sequential": True,
}
```

**NPU 格式** (`ascend_quant_config`)：
```python
{
    "model_quant_type": "W4A16",
    "group_size": 128,
    "quant_layer_types": ["GPTQQuantLinear"],
    "include_g_idx": True,
    "has_offset": True,
}
```
