# Ascend NPU 集成文档

## 1. vLLM-Ascend 集成概览

NPUSlim 通过插件系统扩展 vLLM-Ascend，为华为 Ascend NPU 提供以下增强功能：

- **自定义量化方案**：注册 W4A16、Sparse24 等量化方案到 vLLM-Ascend 的 scheme 注册表。
- **Method Adapter 修复**：修正上游 `AscendLinearMethod.create_weights` 中 per-group 参数维度设置的 bug。
- **MoE 算子扩展**：注册 `AscendZeroExpertFusedMoE` 作为 `ZeroExpertFusedMoE` 的 Ascend OOT（Out-Of-Tree）实现。

插件仅在检测到 NPU 后端且 `vllm_ascend` 包已安装时加载：

```python
# npuslim.plugins.__init__.py
if _load_backend_name() == "npu" and _module_available("vllm_ascend"):
    _register_plugin("npuslim.plugins.vllm_ascend")
```

## 2. 量化方案注册

### 2.1 注册机制

vLLM-Ascend 使用 `@register_scheme` 装饰器注册量化方案。NPUSlim 通过 `register_patch` 的 `registrar` 模式集成该机制：

```python
from vllm_ascend.quantization.methods.registry import register_scheme
from npuslim.plugins.registry import register_patch, package_version_range

@register_patch(
    registrar=register_scheme("W4A16", "linear"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendW4A16LinearMethod(AscendLinearScheme):
    ...
```

与 `target` 模式不同，`registrar` 模式在模块导入时立即执行注册，而非等到 `apply_all_patches()` 调用。这确保 vLLM-Ascend 的 scheme 注册表在推理引擎初始化前已完成填充。

### 2.2 W4A16 量化方案

**位置**：`src/npuslim/plugins/vllm_ascend/quantization/methods/w4a16_linear.py`

W4A16 方案使用 4-bit 量化权重与 16-bit 激活，支持 per-channel 和 per-group 两种量化模式。

**权重格式（NPUSlim 列式打包）**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `weight` | `[output_size, input_size // 8]` | int32 打包的 int4 权重，沿 input 维度打包 |
| `weight_scale` | `[output_size, num_groups]` | per-group 缩放因子 |
| `weight_offset` | `[output_size, num_groups]` | per-group 偏移量 |

**处理流程**（`process_weights_after_loading`）：

1. 使用 `unpack_from_int32()` 将 int4 从 int32 中解包
2. 转置权重 `[N, K] → [K, N]`，适配 NPU API
3. 转置缩放因子 `[N, num_groups] → [num_groups, N]`
4. 可选的 NZ 格式转换（`maybe_trans_nz`）

**推理计算**（`apply`）：

```python
def apply(self, layer, x, bias=None, tp_rank=0):
    scale = layer.weight_scale.to(x.dtype)
    offset = layer.weight_offset.to(x.dtype)

    if use_per_group:
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x, weight=layer.weight, antiquant_scale=scale,
            antiquant_offset=offset, bias=bias,
            antiquant_group_size=layer.group_size,
        )
    else:
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x, weight=layer.weight, antiquant_scale=scale,
            antiquant_offset=offset, bias=bias,
        )
```

### 2.3 Sparse24 稀疏方案

**位置**：`src/npuslim/plugins/vllm_ascend/quantization/methods/sparse24_linear.py`

Sparse24 方案实现 2:4 结构化稀疏推理，使用 AscendC `sparse_matmul_4to2` 内核。

**权重格式**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `weight` | `[output_size, input_size // 2]` | int8 密化非零元素 |
| `weight_scale` | `[output_size]` | float16 per-channel 对称缩放 |
| `weight_index` | 1D uint8 | AscendC tiled index |

**推理计算**：

```python
def apply(self, layer, x, bias=None, tp_rank=0):
    # 动态量化输入到 int8
    max_val = x_2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    x_scale = max_val / 127.0
    x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

    # AscendC 稀疏矩阵乘法
    c_int32 = sparse_matmul_4to2(x_int8, layer.weight, layer.weight_index)
    out = c_int32.float() * (x_scale * layer.weight_scale.unsqueeze(0))
    return out.to(x.dtype)
```

Sparse24 方案通过 `get_param_extra_attrs` 为 `weight_index` 参数注入自定义 stacked weight loader，支持将独立分片的 tiled index 合并到 fused stacked buffer。

## 3. Method Adapter 补丁

**位置**：`src/npuslim/plugins/vllm_ascend/quantization/method_adapters.py`

### 3.1 问题描述

上游 `vllm_ascend/quantization/method_adapters.py` 中 `AscendLinearMethod.create_weights` 存在两个问题：

1. **per-group 参数维度 bug**：仅为 `weight_scale_second` / `weight_offset_second` 设置 `input_dim=1`，遗漏了常规 `weight_scale` / `weight_offset`。对于 `RowParallelLinear`（如 `o_proj`、`down_proj`），per-group 参数依赖 `input_size`，必须设置 `input_dim=1` 以确保张量并行分片正确。

2. **缺少参数属性传播**：未将 `quant_method` 上的 `get_param_extra_attrs` 返回值应用到创建的参数上。

### 3.2 补丁逻辑

```python
@register_patch(
    target="vllm_ascend.quantization.method_adapters",
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
def patch_create_weights(module):
    original_create_weights = module.AscendLinearMethod.create_weights

    def patched_create_weights(self, layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs):
        original_create_weights(...)

        # 修复 1：为 RowParallelLinear 的 per-group 参数设置 input_dim
        if isinstance(layer, RowParallelLinear):
            for name, param in layer.named_parameters(recurse=False):
                if name in ("weight_scale", "weight_offset"):
                    if param.ndim > 1 and param.shape[1] > 1:
                        param.input_dim = 1

        # 修复 2：传播 quant_method 的参数属性
        get_param_extra_attrs = getattr(self.quant_method, "get_param_extra_attrs", None)
        if callable(get_param_extra_attrs):
            for name, param in layer.named_parameters(recurse=False):
                extra_attrs = get_param_extra_attrs(name)
                if extra_attrs:
                    module.set_weight_attrs(param, extra_attrs)

    module.AscendLinearMethod.create_weights = patched_create_weights
```

## 4. NPU 特定算子扩展

### 4.1 AscendZeroExpertFusedMoE

**位置**：`src/npuslim/plugins/vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py`

vLLM-Ascend 为 `FusedMoE` 注册了 OOT 替换，但未覆盖 `ZeroExpertFusedMoE`。LongcatFlash 模型使用 `ZeroExpertFusedMoE`，需要 Ascend 兼容的实现。

**注册方式**：

```python
@register_patch(
    registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendZeroExpertFusedMoE(ZeroExpertFusedMoE, AscendFusedMoE):
    ...
```

**核心设计**：该类同时继承 `ZeroExpertFusedMoE`（零 expert 控制）和 `AscendFusedMoE`（Ascend 推理路径），提供两种执行模式：

| 模式 | 条件 | 执行流程 |
|------|------|---------|
| EP（MC2/All2All） | `moe_comm_type in {MC2, FUSED_MC2, ALLTOALL}` | `prepare → route → zero-expert filter → apply → finalize` |
| 非 EP（AllGather） | 其他 | `route → zero-expert filter → AscendFusedMoE.forward` |

EP 模式下路由在 `prepare` 之后计算，确保 memoized top-k weights/ids 与 prepared token 布局一致。非 EP 模式下 `prepare` 不改变 token 维度，可直接使用上游 memoization 机制。

## 5. 配置和使用方式

### 5.1 安装

```bash
# 安装 NPUSlim 及 NPU 依赖
pip install npuslim[npu]
```

确保 `ASCEND_HOME_PATH` 环境变量已设置，且 `torch_npu` 可用。

### 5.2 量化配置

在 vLLM 的量化配置中使用 NPUSlim 注册的方案：

```json
{
  "quantization_config": {
    "quant_method": "W4A16",
    "group_size": 128
  }
}
```

### 5.3 版本兼容性

| vllm-ascend 版本 | 支持状态 |
|-----------------|---------|
| < 0.19 | 使用 `global_num_experts` 参数 |
| 0.19 ~ 0.20.1 | 完全支持 |
| >= 0.20.1 | patch 自动跳过，需确认上游原生支持 |

### 5.4 排查指南

确认 NPU 插件是否加载：

```
[NPUSlimPatch] Applied registrar: AscendW4A16LinearMethod
[NPUSlimPatch] Applied patch: patch_create_weights -> vllm_ascend.quantization.method_adapters
[NPUSlimPatch] Registered NPUSlim with vLLM-Ascend
```

如果未看到上述日志，检查：
1. `ASCEND_HOME_PATH` 环境变量是否设置
2. `torch_npu` 是否可导入
3. vLLM-Ascend 版本是否在支持范围内
