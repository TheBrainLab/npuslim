# Transformers 集成文档

## 1. 集成概览

NPUSlim 通过 HuggingFace Transformers 的量化器扩展机制，注册自定义量化器以支持加载 NPUSlim 量化的模型。当前提供两个量化器：

| 量化器 | 量化方法 | 状态 |
|--------|---------|------|
| `QuipHfQuantizer` | QuIP（4-bit 量化） | 活跃 |
| `Sparse24HfQuantizer` | 2:4 结构化稀疏 | 已禁用（等待 Transformers 原生支持） |

### 注册机制

Transformers 量化器通过两种机制注册：

1. **Entry Points**（自动注册）：在 `pyproject.toml` 中声明，Transformers 在加载量化模型时按需加载。

```toml
[project.entry-points."transformers.quantizers"]
quip = "npuslim.plugins.transformers.quantizers.quantizer_quip:QuipHfQuantizer"
sparse24 = "npuslim.plugins.transformers.quantizers.quantizer_sparse24:Sparse24HfQuantizer"
```

2. **register_patch registrar 模式**（运行时注册）：使用 Transformers 的 `register_quantization_config` 和 `register_quantizer` API 直接注册到框架内部注册表。

```python
from transformers.quantizers import register_quantization_config, register_quantizer
from npuslim.plugins.registry import register_patch, package_version_range

@register_patch(
    registrar=register_quantization_config("quip"),
    condition=package_version_range("transformers", max_version="4.58.0"),
)
class QuipConfig(QuantizationConfigMixin):
    ...
```

两种机制同时存在以确保兼容性：Entry Points 提供 fallback 加载路径，registrar 模式确保在 NPUSlim 插件初始化后立即可用。

## 2. QuIP 量化器

**位置**：`src/npuslim/plugins/transformers/quantizers/quantizer_quip.py`

### 2.1 概述

QuIP（Quantization with Incoherence Processing）是一种基于随机正交变换的 4-bit 量化方法。`QuipHfQuantizer` 实现 HuggingFace 的 `HfQuantizer` 接口，使 Transformers 能够自动加载 NPUSlim 产出的 QuIP 量化模型。

### 2.2 配置类

```python
@dataclass
class QuipConfig(QuantizationConfigMixin):
    bits: int = 4                    # 量化位数
    quant_func: str = "rms"          # 量化函数："rms" 或 "minmax"
    preproc_proj_mode: int = 2       # 预处理投影模式（2 = Butterfly）
    checkpoint_format: str = "quip"  # 检查点格式
```

配置保存在模型的 `config.json` 中，Transformers 通过 `quant_method` 字段选择对应的量化器。

### 2.3 量化器实现

`QuipHfQuantizer` 继承 `HfQuantizer`，实现以下关键方法：

**validate_environment**：验证 `safetensors` 包是否可用。

**_process_model_before_weight_loading**：在权重加载前将 `nn.Linear` 替换为 `QuIPLinear`。

```python
def _process_model_before_weight_loading(self, model, **kwargs):
    self._replace_with_quip_linear(model)
    return model
```

替换逻辑遍历模型中 `layers` 路径下的所有 `nn.Linear`，使用原始线性层的维度创建 `QuIPLinear`：

```python
def _replace_with_quip_linear(self, model):
    from npuslim.algorithms.quantization.quip.quip_algo import QuIPLinear

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if "layers" not in name:
            continue

        quip_linear = QuIPLinear(
            bits=self.quantization_config.bits,
            infeatures=module.in_features,
            outfeatures=module.out_features,
            has_zero=(self.quantization_config.quant_func == "minmax"),
            bias=module.bias is not None,
            proj_mode=self.quantization_config.preproc_proj_mode,
        )
        setattr(parent, child_name, quip_linear)
```

**_process_model_after_weight_loading**：无需后处理，权重通过 `from_pretrained` 机制直接加载到 `QuIPLinear` 的缓冲区中。

### 2.4 模型配置格式

QuIP 量化模型的 `config.json` 中需要包含以下字段：

```json
{
  "quantization_config": {
    "quant_method": "quip",
    "bits": 4,
    "quant_func": "rms",
    "preproc_proj_mode": 2,
    "checkpoint_format": "quip"
  }
}
```

## 3. Sparse24 量化器

**位置**：`src/npuslim/plugins/transformers/quantizers/quantizer_sparse24.py`

### 3.1 概述

`Sparse24HfQuantizer` 为 2:4 结构化稀疏模型提供 HuggingFace 加载支持。该量化器将 `nn.Linear` 替换为 `AscendSparse24Linear`，使打包的稀疏张量（`weight`、`weight_scale`、`weight_index`）能够直接从 safetensors 检查点加载。

### 3.2 当前状态

该量化器当前通过 `always_disable` 条件禁用：

```python
@register_patch(
    registrar=register_quantization_config("sparse24"),
    condition=always_disable,  # 禁用，等待 Transformers 原生支持
)
class Sparse24Config(QuantizationConfigMixin):
    ...
```

禁用原因：Transformers 目前不支持 Ascend NPU 稀疏推理。当框架添加原生支持后，将 `condition` 修改为 `package_version_range("transformers", max_version="...")` 即可启用。

### 3.3 配置类

```python
@dataclass
class Sparse24Config(QuantizationConfigMixin):
    sparsity_type: str = "2:4"  # 稀疏模式
```

### 3.4 量化器实现

`Sparse24HfQuantizer` 的结构与 `QuipHfQuantizer` 类似：

```python
class Sparse24HfQuantizer(HfQuantizer):
    requires_calibration = False
    required_packages = ["safetensors"]

    def _replace_with_sparse24_linear(self, model):
        from npuslim.algorithms.quantization.sparsegpt.sparsegpt_algo import AscendSparse24Linear

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if "layers" not in name:
                continue

            sparse_linear = AscendSparse24Linear(
                infeatures=module.in_features,
                outfeatures=module.out_features,
                bias=module.bias is not None,
            )
            setattr(parent, child_name, sparse_linear)
```

### 3.5 模型配置格式（未来启用后使用）

```json
{
  "quantization_config": {
    "quant_method": "sparse24",
    "sparsity_type": "2:4"
  }
}
```

## 4. 配置和使用方式

### 4.1 安装

```bash
pip install npuslim
```

Transformers 量化器通过 entry_points 自动注册，无需额外配置。

### 4.2 加载量化模型

```python
from transformers import AutoModelForCausalLM

# 加载 QuIP 量化模型
model = AutoModelForCausalLM.from_pretrained(
    "path/to/quip-quantized-model",
    device_map="auto",
)
# Transformers 自动检测 config.json 中的 quant_method="quip"
# 并通过 QuipHfQuantizer 处理模型加载
```

### 4.3 版本兼容性

| Transformers 版本 | 支持状态 |
|-------------------|---------|
| < 4.58.0 | 完全支持，registrar 模式生效 |
| >= 4.58.0 | 通过 entry_points fallback 加载 |

### 4.4 直接访问量化器

如需在代码中直接使用量化器类：

```python
from npuslim.plugins.transformers import QuipConfig, QuipHfQuantizer

config = QuipConfig(bits=4, quant_func="rms")
quantizer = QuipHfQuantizer(config)
```

### 4.5 排查指南

如果量化模型加载失败：

1. 检查 `config.json` 中 `quantization_config.quant_method` 是否为 `"quip"`
2. 确认 `safetensors` 已安装
3. 检查 NPUSlim 是否正确安装：`pip show npuslim`
4. 查看加载日志中是否出现 `[NPUSlimPatch]` 相关的注册信息
