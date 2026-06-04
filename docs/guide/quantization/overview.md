# 量化算法总览

## 为什么需要量化

大语言模型（LLM）的参数量通常达到数十亿甚至数千亿级别，原始的 FP16/BF16 权重对显存和带宽构成巨大压力。**量化**（Quantization）通过将高精度浮点权重映射到低比特整数表示，能够在尽量保持模型精度的前提下，显著降低：

- **显存占用**：4-bit 量化可将模型体积压缩至原始的 25%
- **推理延迟**：低比特矩阵乘法的计算吞吐量更高
- **部署成本**：降低硬件门槛，使模型可在消费级 GPU 或 NPU 上运行

NPUSlim v2 提供了多种业界领先的量化和稀疏化算法，支持 GPU（CUDA）和 NPU（华为昇腾 Ascend）后端。

## 算法一览表

| 算法 | 注册名 | 别名 | 类型 | 比特数 | 需要校准数据 | 典型场景 |
|------|--------|------|------|--------|------------|---------|
| GPTQ | `GPTQ` | `gptq`, `GPTQStepwise` | 权重量化 | 2/3/4/8 | 是 | W4A16 高精度压缩 |
| INT8Dynamic | `INT8Dynamic` | `INT8Dyn`, `int8_dyn` | 权重+激活量化 | 8 | 否（可选） | W8A8 部署、NPU 推理 |
| QuIP | `QuIP` | `quip` | 权重量化 | 2/3/4/8 | 是 | W4A16 + 不相干性增强 |
| SparseGPT | `SparseGPT` | `sparsegpt`, `sparse_gpt` | 稀疏化 | — | 是 | 2:4 结构化稀疏 / 非结构化稀疏 |

## 算法分类

### 按技术路线分类

```
量化算法
├── Hessian 类（需要二阶统计信息）
│   ├── GPTQ      -- 基于 Hessian 的逐列权重量化 + 误差补偿
│   ├── QuIP      -- 不相干性处理 + LDLQ 舍入
│   └── SparseGPT -- 基于 Hessian 的逐列剪枝 + 误差补偿
└── Observer 类（基于统计观察器）
    └── INT8Dynamic -- per-channel/per-tensor/per-group 观察器
```

### 按精度损失与压缩率分类

| 压缩比 | 推荐算法 | 精度损失 |
|--------|---------|---------|
| 2x（W8A8） | INT8Dynamic | 极低 |
| 4x（W4A16） | GPTQ / QuIP | 低~中 |
| 2x 稀疏（50% 非结构化） | SparseGPT | 低 |
| 2x 稀疏（2:4 结构化） | SparseGPT | 低 |

## Hessian 公共逻辑

GPTQ、QuIP 和 SparseGPT 三种算法共享同一套 Hessian 运行时基础设施，定义在 `BaseHessianAlgorithm`（`algorithms/quantization/hessian/base_hessian_algo.py`）和 `BaseHessianModule`（`algorithms/quantization/hessian/hessian_common.py`）中。

### Hessian 累积机制

`BaseHessianModule.add_batch()` 在前向传播过程中通过注册 hook 逐层累积 Fisher 信息矩阵的近似：

```python
# 简化的 Hessian 累积逻辑
def add_batch(self, inp, out):
    # inp: [batch, seq_len, hidden_dim] -> reshape -> [hidden_dim, batch*seq_len]
    inp = inp.reshape((-1, inp.shape[-1])).t()
    self.H *= self.nsamples / (self.nsamples + batch)
    self.nsamples += batch
    inp = sqrt(2 / self.nsamples) * inp
    self.H += inp @ inp.t()   # 累积二阶统计量
```

### Hessian 求逆与阻尼

`compute_hinv()` 对累积的 Hessian 矩阵执行 Cholesky 分解并求逆。当矩阵病态时，自动递增阻尼系数 `percdamp`（默认 0.01）直至分解成功：

```python
def compute_hinv(self, hessian):
    current_percdamp = 0.0
    while current_percdamp < 1.0:
        try:
            h_try = hessian.clone()
            h_try[diag_idx, diag_idx] += damp   # 阻尼正则化
            chol = torch.linalg.cholesky(h_try)
            inv_chol = torch.cholesky_inverse(chol)
            hinv = torch.linalg.cholesky(inv_chol, upper=True)
            break
        except LinAlgError:
            current_percdamp += step     # 增大阻尼重试
```

### 逐层流式处理流程

`BaseHessianAlgorithm.process_chunk()` 实现了统一的 chunk 处理生命周期：

1. **捕获初始输入**：在第一个 chunk 中，通过 hook 捕获校准数据经 embedding 层后的输出作为第一层的输入
2. **逐层迭代**：
   - 将当前层权重加载到运行时模型
   - 提取 Linear 层目标，创建算法特定的 handler（如 `GPTQModule`）
   - **校准**：前向传播校准数据，handler 通过 hook 累积 Hessian
   - **量化/剪枝**：调用 handler 的 `fasterquant()`/`fasterprune()` 方法
   - **前向传播**：将处理后的层输出作为下一层的输入
3. **释放资源**：卸载运行时模型中的权重张量

```
chunk 处理流程:
┌─────────────────────────────────────────────┐
│  捕获初始输入 (仅第一个 chunk)               │
├─────────────────────────────────────────────┤
│  for layer in chunk.layers:                 │
│    ├─ 加载权重到运行时模型                    │
│    ├─ 创建 handler (GPTQModule / etc.)      │
│    ├─ 校准: 累积 Hessian                    │
│    ├─ 量化/剪枝: fasterquant / fasterprune   │
│    ├─ 前向传播: 计算下一层输入                │
│    └─ 卸载权重                               │
├─────────────────────────────────────────────┤
│  更新 chunk 元数据 (tensor_types)            │
└─────────────────────────────────────────────┘
```

## BaseQuantizationAlgorithm 基类

所有量化算法（包括 Hessian 类和 Observer 类）均继承自 `BaseQuantizationAlgorithm`（`algorithms/quantization/base_quant_algo.py`），提供以下通用机制：

### 运行时上下文设置

```python
def set_runtime_context(
    self,
    *,
    model_obj=None,          # 模型对象 (BaseLLMModel 实例)
    model_config=None,       # 模型配置 (transformers.PretrainedConfig)
    skip_layer_names=None,   # 跳过量化的层名列表
):
```

### 层名跳过匹配

支持三种匹配模式：

```python
# 精确匹配
ignore_layers: ["lm_head"]

# glob 通配符
ignore_layers: ["model.layers.*.mlp.down_proj"]

# 正则表达式 (以 "re:" 前缀)
ignore_layers: ["re:.*\\.layernorm\\..*"]
```

匹配规则：
- `re:pattern` -- 正则全匹配（`re.fullmatch`）
- 精确名称匹配或前缀匹配（`name == skip` 或 `name.startswith(skip + ".")`）
- `fnmatch` glob 匹配

### 后端感知

`target_backend` 属性决定了量化结果的打包格式：

- **GPU (CUDA)**：使用 AutoGPTQ 兼容的 `qweight`/`qzeros`/`scales`/`g_idx` 张量布局
- **NPU (Ascend)**：使用 Ascend 原生的 `weight`/`weight_scale`/`weight_offset` 布局

可通过配置中的 `save_backend` 参数覆盖自动检测的后端。

## 如何选择算法

### 决策流程

```
是否需要量化？
├─ 仅需推理加速，精度要求极高
│   └─ INT8Dynamic (W8A8, 几乎无损)
├─ 需要大幅压缩 (4x)
│   ├─ 追求最佳精度
│   │   └─ QuIP (不相干性处理 + LDLQ)
│   └─ 追求通用性和生态兼容
│       └─ GPTQ (vLLM/AutoGPTQ 生态成熟)
└─ 需要结构化稀疏 (NPU 硬件加速)
    └─ SparseGPT (2:4 结构化稀疏)
```

### 按后端选择

| 后端 | 推荐算法 | 说明 |
|------|---------|------|
| CUDA GPU | GPTQ, INT8Dynamic | GPU 上 GPTQ 生态最成熟 |
| Ascend NPU | INT8Dynamic, SparseGPT | NPU 原生支持 W8A8 和 2:4 稀疏 |
| CPU | INT8Dynamic | INT8Dynamic 无需校准数据，最轻量 |

### 按模型规模选择

| 模型规模 | 推荐方案 | 理由 |
|---------|---------|------|
| < 1B | INT8Dynamic | 小模型对量化敏感度低，无需复杂算法 |
| 1B ~ 14B | GPTQ W4A16 | 平衡压缩率与精度 |
| 14B ~ 70B | GPTQ W4A16 / QuIP W4A16 | 大模型量化鲁棒性好 |
| MoE 模型 | GPTQ / SparseGPT | MoE 模型参数量大，稀疏化收益高 |
