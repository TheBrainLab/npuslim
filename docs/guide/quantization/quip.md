# QuIP 量化算法

## 算法原理

QuIP（Quantization with Incoherence Processing）是一种通过**不相干性处理**（Incoherence Processing）增强的低比特权重量化算法，由 Chee 等人在 2023 年提出。其核心洞察是：当权重矩阵和 Hessian 矩阵的"重要性"分布越均匀时，量化误差越小。

### 不相干性处理

QuIP 在量化前对权重施加两种预处理操作：

1. **对角重缩放（Diagonal Rescaling）**：平衡权重和 Hessian 的尺度差异

   $$\text{scaleWH} = \left(\frac{\text{diag}(H)}{\text{diag}(W^T W)}\right)^{1/4}$$

   将权重乘以 `scaleWH`，Hessian 除以对应的行/列缩放因子，使两者的对角元素趋于一致。

2. **随机正交投影（Random Orthogonal Projection）**：通过随机正交矩阵将权重和 Hessian 变换到"更均匀"的基上

   $$\hat{W} = U W V^T, \quad \hat{H} = V H V^T$$

   其中 $U$ 和 $V$ 为随机正交矩阵（Butterfly 矩阵或完全随机正交矩阵）。

量化完成后，执行逆变换恢复原始基。

### LDLQ 舍入算法

QuIP 使用 LDLQ（LDL-decomposition-based Quantization）替代简单的最近整数舍入。LDLQ 通过 Hessian 的 LDL 分解（实际实现中使用 Cholesky 分解）计算最优舍入方向，在保持整数约束的前提下最小化 Hessian 加权量化误差。

NPUSlim 支持多种 LDLQ 变体：

| 方法 | 说明 |
|------|------|
| `ldlq` | 标准 LDLQ，逐列反向舍入 |
| `ldlqRG` | 按对角线排序后再执行 LDLQ |
| `allbal` | 全局均衡舍入，多轮迭代 |
| `ldlbal_admm` | 排序 + LDLQ（固定点迭代） |
| `ldl_gptqequiv` | 等价于 GPTQ 的前向舍入方式 |

### 数学直觉

考虑量化单个权重列 $w$ 时对输出的影响：

$$\Delta \text{output} \approx (w - \hat{w}) \cdot x \sim \|w - \hat{w}\|_H$$

不相干性处理的目标是让 Hessian 的对角线尽可能均匀，从而使得每个坐标的量化误差贡献均衡，避免某些"关键"坐标的量化误差主导整体损失。

## 实现细节

### 整体架构

| 组件 | 位置 | 职责 |
|------|------|------|
| `QuIPAlgorithm` | `algorithms/quantization/quip/quip_algo.py` | 顶层算法类 |
| `QuIPModule` | 同上 | 逐层 QuIP 处理，包含 `preproc()` / `fasterquant()` / `postproc()` |
| `QuIPLinear` | 同上 | 量化权重打包与推理时的反量化 |
| `LDLQConfig` | 同上 | LDLQ 舍入算法配置 |
| Butterfly 矩阵工具 | 同上 | 生成和操作随机正交投影矩阵 |

### 处理流程

```
QuIPModule 处理流程:
┌─────────────────────────────────────────┐
│  1. Hessian 累积 (继承自 BaseHessianModule) │
├─────────────────────────────────────────┤
│  2. preproc():                          │
│     a. 对角重缩放 (scaleWH)             │
│     b. 随机正交投影 (U, V 矩阵)          │
│     c. Hessian 预处理 (dead rows, damp) │
├─────────────────────────────────────────┤
│  3. fasterquant():                      │
│     a. 计算 scale / zero (minmax 模式)  │
│        或 scale_rms (rms 模式)          │
│     b. LDLQ 舍入 (调用 _round_ldlq)     │
│     c. 收集量化参数                      │
├─────────────────────────────────────────┤
│  4. postproc():                         │
│     a. 逆正交投影 (U^T W V)             │
│     b. 逆对角重缩放                      │
├─────────────────────────────────────────┤
│  5. 释放 Hessian 和投影矩阵             │
└─────────────────────────────────────────┘
```

### 正交投影矩阵

NPUSlim 支持四种投影矩阵生成模式（`preproc_proj_mode`）：

| 模式值 | 名称 | 说明 |
|--------|------|------|
| 0 | `butterfly_permute` | Butterfly 矩阵 + 随机排列 |
| 1 | `butterfly_permute_noblock` | Butterfly 矩阵 + 排列（无分块） |
| 2 | `butterfly_nopermute` | Butterfly 矩阵（无排列），**默认** |
| 3 | `random_ortho` | 完全随机正交矩阵（Scipy `special_ortho_group`） |

Butterfly 矩阵通过将维度分解为质因数，在每个质因数维度上应用随机 2x2 正交变换来构建，计算成本远低于完全随机正交矩阵，同时仍能有效打乱权重的"重要性"分布。

投影矩阵的随机种子保存在模型中（`proj_seed_u`、`proj_seed_v`），推理时根据种子重新生成矩阵执行反量化。

### 权重打包与推理

`QuIPLinear` 存储以下张量：

| 张量 | 形状 | 说明 |
|------|------|------|
| `qweight` | `[infeatures // 32 * bits, outfeatures]` | 打包的整数权重 |
| `scales` | `[outfeatures, 1]`（minmax）或 `[1]`（rms） | 缩放因子 |
| `zeros` | `[outfeatures, 1]`（仅 minmax） | 零点 |
| `scaleWH` | `[infeatures]` | 对角重缩放因子 |
| `proj_seed_u` | 标量 | 投影矩阵 U 的随机种子 |
| `proj_seed_v` | 标量 | 投影矩阵 V 的随机种子 |

推理时的反量化过程：

```python
def forward(self, x):
    weight_int = self._unpack_weights()      # 解包整数权重
    w = self._dequantize(weight_int)          # 反量化为浮点
    w = self._postproc(w)                     # 逆投影 + 逆缩放
    return F.linear(x, w, self.bias)
```

## 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wbits` | int | 4 | 量化比特数，支持 2/3/4/8 |
| `quant_func` | str | `rms` | 量化函数：`rms`（RMS 归一化）或 `minmax`（min-max 缩放） |
| `ldlq_method` | str | `ldlq` | LDLQ 舍入方法 |
| `npasses` | int | 0 | 贪心迭代轮数（`allbal` 和部分 LDLQ 变体使用） |
| `unbiased` | bool | False | 是否使用无偏舍入（随机舍入而非确定舍入） |
| `blocksize` | int | 128 | LDLQ 块大小 |
| `percdamp` | float | 0.01 | Hessian 阻尼系数 |
| `preproc_hessian` | bool | True | 是否预处理 Hessian |
| `preproc_rescale` | bool | True | 是否执行对角重缩放 |
| `preproc_proj` | bool | True | 是否执行随机正交投影 |
| `preproc_proj_mode` | int | 2 | 投影矩阵生成模式 |
| `incoh_processing` | bool | True | 启用完整的不相干性处理流水线 |
| `fake_quant` | bool | False | 伪量化模式 |
| `max_calib_samples` | int | 128 | 最大校准样本数 |

### incoh_processing 快捷参数

当设置 `incoh_processing: true` 时，会自动启用完整的不相干性处理流水线：

```python
if incoh_processing:
    quant_func = "rms"
    preproc_hessian = True
    preproc_rescale = True
    preproc_proj = True
```

这等价于手动设置所有预处理选项。在大多数场景下，建议保持 `incoh_processing: true`。

## 完整 YAML 配置示例

### 标准不相干性处理（推荐）

```yaml
metadata:
  name: "QuIP_W4A16_Recipe"
  description: "QuIP 4-bit quantization with incoherence processing"

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
  - name: "QuIP_Quantization"
    type: compressor
    model: "@model"
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
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 手动控制预处理步骤

```yaml
algorithm:
  type: QuIP
  wbits: 4
  quant_func: rms
  ldlq_method: ldlq
  incoh_processing: false       # 关闭快捷设置，手动控制
  preproc_hessian: true
  preproc_rescale: true
  preproc_proj: true
  preproc_proj_mode: 3           # 使用完全随机正交矩阵
  percdamp: 0.01
```

### minmax 量化模式

```yaml
algorithm:
  type: QuIP
  wbits: 4
  quant_func: minmax             # 使用 min-max 缩放替代 RMS
  ldlq_method: ldlq
  incoh_processing: false
  preproc_hessian: true
  preproc_rescale: false         # 关闭对角重缩放
  preproc_proj: false            # 关闭正交投影
```

### NPU 部署

```yaml
metadata:
  name: "QuIP_W4A16_NPU_Recipe"
  description: "QuIP for Ascend NPU"

resources:
  - id: model
    type: Qwen3Model
    path: Qwen/Qwen3-0.6B
    device_map: npu

  - id: calib_data
    type: C4Dataset
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "QuIP_Quantization"
    type: compressor
    model: "@model"
    dataloader:
      dataset: "@calib_data"
      batch_size: 1
    algorithm:
      type: QuIP
      wbits: 4
      incoh_processing: true
      save_backend: npu
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

## 量化元数据

**GPU 格式** (`quantization_config`)：
```python
{
    "bits": 4,
    "quant_func": "rms",
    "quant_method": "quip",
    "checkpoint_format": "quip",
    "preproc_proj_mode": 2,
}
```

**NPU 格式** (`ascend_quant_config`)：
```python
{
    "model_quant_type": "W4A16",
    "group_size": -1,
    "quant_layer_types": ["QuIPLinear"],
    "include_g_idx": False,
    "has_offset": True,
}
```

## 与 GPTQ 的对比

| 特性 | QuIP | GPTQ |
|------|------|------|
| 预处理 | 不相干性处理（重缩放 + 正交投影） | Activation ordering（对角线排序） |
| 舍入算法 | LDLQ（多种变体） | 最近整数 + 误差补偿 |
| 2-bit 量化 | 支持较好 | 支持，但精度下降明显 |
| 推理开销 | 需要反投影（计算量略高） | 直接解包 |
| 校准数据需求 | 需要 | 需要 |
| 理论最优性 | 在不相干性假设下可证明逼近最优 | 贪心近似 |

QuIP 在 2-bit 和 3-bit 等极低比特场景下通常优于 GPTQ，但推理时需要额外的反投影计算。
