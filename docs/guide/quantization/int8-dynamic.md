# INT8 动态量化

## 算法原理

INT8 动态量化（INT8 Dynamic Quantization）是一种轻量级的量化方案，将权重量化为 8-bit 整数，激活值在推理时动态量化为 8-bit。与 GPTQ 等 Hessian 类算法不同，INT8 动态量化**不需要校准数据**（也可使用校准数据进行更精确的统计），也不需要二阶信息。

### 权重量化策略

NPUSlim 的 INT8 动态量化支持三种权重缩放策略：

| 策略 | 注册键 | 说明 |
|------|--------|------|
| per-tensor | `per-tensor` | 整个权重矩阵使用单一缩放因子 |
| per-channel | `per-channel` | 每个输出通道独立缩放（默认，推荐） |
| per-group | `per-group` | 按固定大小分组缩放 |

### 量化公式

权重量化采用对称绝对值最大（abs-max）方案：

$$\text{scale} = \max(|W|)$$
$$W_q = \text{round}\left(\frac{W}{\text{scale} / (2^{b-1} - 1)}\right), \quad W_q \in [-2^{b-1}, 2^{b-1}-1]$$

其中 $b = 8$，缩放因子存储为 `weight_scale` 张量，与量化权重配对保存。

### 激活量化

激活量化在推理时按 token 动态执行（per-token），不需要离线校准。这意味着激活的缩放因子不保存在模型中，而是由推理引擎实时计算。

### 与 GPTQ 的区别

| 特性 | INT8Dynamic | GPTQ |
|------|-------------|------|
| 权重比特数 | 8 | 2/3/4/8 |
| 激活量化 | 动态 per-token | 不量化（A16） |
| 是否需要校准数据 | 否（可选） | 是 |
| 是否使用 Hessian | 否 | 是 |
| 计算复杂度 | 低 | 高（Hessian 累积 + 求逆） |
| 压缩比 | ~2x | ~4x（W4） |
| 精度损失 | 极低 | 低~中 |
| 推理加速 | 是（W8A8 int8 matmul） | 有限（W4A16 需要反量化） |

## 实现细节

### 整体架构

INT8 动态量化不依赖 Hessian 基础设施，而是基于 Observer 模式：

| 组件 | 位置 | 职责 |
|------|------|------|
| `INT8DynamicAlgorithm` | `algorithms/quantization/int8_dynamic/int8_dynamic_algo.py` | 顶层算法类 |
| `BaseWeightObserver` | 同上 | 观察器基类 |
| `AbsMaxChannelWiseWeightObserver` | 同上 | per-channel abs-max 观察器（默认） |
| `AbsMaxPerTensorWeightObserver` | 同上 | per-tensor abs-max 观察器 |
| `AbsMaxGroupWiseWeightObserver` | 同上 | per-group abs-max 观察器 |
| `PTQObserverHook` | 同上 | 管理 observer 生命周期 |

### 处理流程

```python
def process_chunk(self, chunk):
    # 1. 收集可量化的权重层（跳过 ignore_layers 中的层）
    observer_layers = self._collect_observer_layers(chunk)

    # 2. 为每个权重创建 observer
    for name, tensor in observer_layers.items():
        observer = WeightObserver(quant_bits=8)
        observer(tensor)                 # 观察权重，计算 scale

    # 3. 量化权重
    for name, tensor in observer_layers.items():
        scale = observer.scales()
        quant_weight, stored_scale = quantize_weight_int(
            weight=tensor, scales=scale, bits=8
        )

        # 4. 替换原始权重为量化权重 + 缩放因子
        layer.tensors["xxx.weight"] = quant_weight
        layer.tensors["xxx.weight_scale"] = stored_scale

    # 5. 更新 chunk 元数据（tensor_types）
    chunk.metadata["tensor_types"] = ...
```

### quantize_weight_int 核心逻辑

```python
def quantize_weight_int(weight, scales, bits=8):
    bnt = (1 << (bits - 1)) - 1       # = 127 for int8
    # 扩展 scale 以匹配 weight 形状
    expanded_scale = expand_scales(scales, weight.shape)
    stored_scale = scales / float(bnt)
    quant_weight = round(weight / (expanded_scale / bnt)).clamp(-bnt-1, bnt)
    return quant_weight, stored_scale
```

## 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wbits` | int | 8 | 权重量化比特数，固定为 8 |
| `w_quant_method` | str | `per-channel` | 权重量化策略：`per-tensor` / `per-channel` / `per-group` |
| `a_quant_method` | str | `per-token` | 激活量化策略：`per-token` |
| `group_size` | int | -1 | 分组大小，仅在 `per-group` 模式下有效 |
| `weight_observer` | str | None | 显式指定观察器类型，覆盖 `w_quant_method` |

### 参数选择建议

- **`w_quant_method`**：`per-channel` 是最通用的选择，适合绝大多数场景。`per-tensor` 精度略低但存储更紧凑。`per-group` 提供更细粒度的控制
- **`group_size`**：仅在 `w_quant_method: per-group` 时需要设置，推荐值 128
- **`a_quant_method`**：通常保持 `per-token` 不变

## 完整 YAML 配置示例

### GPU INT8 W8A8（无需校准数据）

```yaml
metadata:
  name: "INT8_Dynamic_Recipe"
  description: "INT8 dynamic quantization for Qwen3-8B on GPU"

resources:
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    device_map: cuda

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"
    algorithm:
      type: INT8Dynamic
      wbits: 8
      w_quant_method: per-channel
      a_quant_method: per-token
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### NPU INT8 W8A8

```yaml
metadata:
  name: "INT8_Dynamic_NPU_Recipe"
  description: "INT8 dynamic quantization for Ascend NPU"

resources:
  - id: qwen3
    type: Qwen3Model
    path: Qwen/Qwen3-8B
    device_map: npu

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"
    algorithm:
      type: INT8Dynamic
      wbits: 8
      w_quant_method: per-channel
    ignore_layers: []
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### 带校准数据的 INT8

```yaml
metadata:
  name: "INT8_Calibrated_Recipe"
  description: "INT8 quantization with calibration dataset"

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
  - name: "INT8_Quantization"
    type: compressor
    model: "@model"
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

## GPU vs NPU 差异

### 输出张量格式

**GPU (CUDA)**：
```
model.layers.0.self_attn.q_proj.weight        # int8 量化权重
model.layers.0.self_attn.q_proj.weight_scale  # FP16 缩放因子
```

**NPU (Ascend)**：
```
model.layers.0.self_attn.q_proj.weight        # int8 量化权重
model.layers.0.self_attn.q_proj.weight_scale  # BF16 缩放因子
```

### 量化元数据

**GPU 格式** (`quantization_config`，遵循 `compressed-tensors` 格式）：
```python
{
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "strategy": "channel",
                "dynamic": False,
                "type": "int",
            },
            "input_activations": {
                "num_bits": 8,
                "strategy": "token",
                "dynamic": True,
                "type": "int",
            },
        }
    },
}
```

**NPU 格式** (`ascend_quant_config`）：
```python
{
    "model_quant_type": "W8A8_DYNAMIC",
    "group_size": -1,
    "quant_layer_types": ["Linear"],
    "include_g_idx": False,
    "has_offset": False,
}
```

### 推理部署

**GPU**：使用 vLLM 的 `compressed-tensors` 量化后端加载量化模型：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./outputs \
    --quantization compressed-tensors
```

**NPU**：通过 vLLM-Ascend 部署，框架自动读取 `quant_model_description.json`：

```bash
bash tools/serve/deploy_vllm.sh ./outputs -d 0 -t 1 -q
```

## ignore_layers 配置

```yaml
# 跳过最后的 lm_head 层
ignore_layers:
  - "lm_head"

# 跳过所有 MLP 的 down_proj
ignore_layers:
  - "model.layers.*.mlp.down_proj"
```

跳过的层保留原始浮点精度，不会被量化。
