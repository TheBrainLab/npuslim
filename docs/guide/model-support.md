# 模型适配指南

本文档详细说明 NPUSlim v2 的模型适配机制，包括基类设计、已支持模型、以及接入新模型的完整流程。

---

## 1 BaseLLMModel 基类设计

`BaseLLMModel`（`src/npuslim/models/base_model.py`）是所有模型适配器的抽象基类，负责封装 tokenizer、config、模型加载/释放等通用逻辑。

### 1.1 构造函数与参数传递

```python
class BaseLLMModel(ABC):
    def __init__(
        self,
        *,
        path: str,                    # 模型路径（本地目录或 Hub repo_id）
        model_hub: str = "hf",        # 模型来源：hf（HuggingFace）或 ms（ModelScope）
        model_kwargs: dict = None,    # 传递给 from_pretrained 的参数
        tokenizer_kwargs: dict = None,
        **kwargs,
    ):
```

构造函数将 `device_map`、`torch_dtype`、`trust_remote_code` 等关键字参数自动分发到 `model_kwargs` 或 `tokenizer_kwargs`，通过 `passthrough_model_keys` / `passthrough_tokenizer_keys` 两个白名单控制。

### 1.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `path` | `Path` | 模型本地路径或 repo_id |
| `model_hub` | `str` | `"hf"` 或 `"ms"` |
| `block_name` | `str` | Transformer 层的命名前缀，如 `"model.layers"` |
| `pre_transformer_module_names` | `list[str]` | Transformer 块之前的模块名（embedding 等） |
| `post_transformer_module_names` | `list[str]` | Transformer 块之后的模块名（norm、lm_head 等） |
| `skip_layer_names` | `list[str]` | 默认跳过不量化的层名（支持 glob/regex） |
| `model_type` | `str` | `"LLM"` 或 `"VLM"` |

### 1.3 核心方法

#### `prepare_metadata()`
加载 tokenizer、processor（可选）和 AutoConfig。该方法在 `__init__` 中自动调用，仅加载轻量元数据，不加载模型权重。对于远程模型，会通过 `model_hub` 参数自动选择 HuggingFace Hub 或 ModelScope 的 `snapshot_download` 来解析本地缓存路径。

#### `prepare_empty_model()`
利用 `accelerate.init_empty_weights()` 在 meta device 上构建模型骨架，不分配实际内存。用于需要模型结构信息但不需要权重的场景（如校准推理前的模型结构分析）。

```python
def prepare_empty_model(self):
    with init_empty_weights():
        self.empty_model = model_cls.from_config(self.config, ...)
    self.empty_model.eval()
    return self.empty_model
```

#### `prepare_full_model()` / `release_full_model()`
完整加载模型到内存/显存，以及释放资源。`prepare_full_model` 调用 `from_pretrained` 加载全部权重。

#### `get_model_loader_candidates()` / `get_tokenizer_loader_candidates()` / `get_processor_loader_candidates()`
返回加载器类的候选列表（按优先级排序）。默认返回 `["AutoModelForCausalLM"]` 和 `["AutoTokenizer"]`，processor 默认为空列表。子类可覆写以指定具体的模型类，例如 VL 模型需要 `AutoModelForImageTextToText`。

### 1.4 model_hub 支持

通过 `get_hub_class(model_hub, class_name)` 工具函数实现：

- `model_hub="hf"`：从 `transformers` 包导入对应类
- `model_hub="ms"`：优先从 `modelscope` 包导入，若失败则回退到 `transformers`

### 1.5 device_map 参数

`device_map` 参数通过 `model_kwargs` 传递给 `from_pretrained`。支持以下值：

- `"cpu"` / `"cuda"` / `"npu"`：直接指定设备
- `"auto"` / `"balanced"` 等：由 accelerate 自动分配
- `dict`：精细控制每层设备映射

---

## 2 已支持模型详解

### 2.1 Qwen3 系列

**注册名**：`Qwen3`（别名：`Qwen3Model`）

**架构特点**：
- 标准 Decoder-Only Transformer 架构
- 同时支持 Dense 模型（如 Qwen3-0.6B、Qwen3-8B）和 MoE 模型（如 Qwen3-235B-A22B）
- MoE 模型含有 `mlp.gate` 路由模块，默认跳过不量化

**层结构配置**：
```python
block_name = "model.layers"
pre_transformer_module_names = ["model.embed_tokens"]
post_transformer_module_names = ["model.norm", "lm_head"]
skip_layer_names = ["lm_head", "model.layers.*.mlp.gate"]
```

**配置示例**：
```yaml
resources:
  - id: qwen3
    type: Qwen3
    path: Qwen/Qwen3-8B
    device_map: cuda
    trust_remote_code: true
    low_cpu_mem_usage: true
```

**注意事项**：MoE 模型的 gate 模块自动加入 `skip_layer_names`，无需用户手动配置。

### 2.2 OPT 系列

**注册名**：`OPT`（别名：`OPTModel`）

**架构特点**：
- Meta 的 Open Pre-trained Transformer，Decoder-Only 架构
- 与 Qwen3 的主要区别在于命名规范：Transformer 层位于 `model.decoder.layers`
- 包含 `embed_tokens` 和 `embed_positions` 两个 embedding 模块

**层结构配置**：
```python
block_name = "model.decoder.layers"
pre_transformer_module_names = ["model.decoder.embed_tokens", "model.decoder.embed_positions"]
post_transformer_module_names = ["model.decoder.final_layer_norm", "lm_head"]
```

**配置示例**：
```yaml
resources:
  - id: opt
    type: OPT
    path: facebook/opt-125m
    device_map: cuda
```

### 2.3 GLM-5

**注册名**：`GLM5`（别名：`Glm5Model`、`GlmMoeDsa`）

**架构特点**：
- 基于 GlmMoeDsa 架构，采用 MLA（Multi-head Latent Attention）注意力机制
- MLA 使用 q_a/q_b 和 kv_a/kv_b 的 LoRA 风格投影
- 混合 Dense/MoE MLP：前 `first_k_dense_replace` 层使用 Dense MLP，其余层使用 256 个路由专家 + 1 个共享专家的 MoE 结构
- 包含 DSA（Dynamic Sparse Attention）索引器子模块

**层结构配置**：
```python
block_name = "model.layers"
pre_transformer_module_names = ["model.embed_tokens"]
post_transformer_module_names = ["model.norm", "lm_head"]
skip_layer_names = ["lm_head", "model.layers.*.mlp.gate"]
```

**配置示例**：
```yaml
resources:
  - id: glm5
    type: GLM5
    path: /data/models/GLM-5
    trust_remote_code: true
    low_cpu_mem_usage: true
    device_map: cpu
```

**注意事项**：
- 必须设置 `trust_remote_code: true`，因为模型使用了自定义代码
- MoE 的 gate 模块自动跳过量化
- MLA 投影层的量化需要特别关注精度影响

### 2.4 Qwen3-VL（多模态）

**注册名**：`Qwen3VL`（别名：`Qwen3VLModel`、`Qwen3VLMoe`、`Qwen3VLMoeModel`）

**架构特点**：
- 视觉-语言多模态模型，包含 Visual Encoder 和 Language Model 两部分
- `model_type` 设置为 `"VLM"`，区别于纯文本模型
- 语言模型路径为 `model.language_model.layers`
- 同时支持 Dense 和 MoE 两种变体（通过 `qwen3_vl` / `qwen3_vl_moe` 区分）
- Visual 分支和 embedding 层默认跳过量化，仅对语言模型层进行量化

**层结构配置**：
```python
model_type = "VLM"
block_name = "model.language_model.layers"
pre_transformer_module_names = ["model.visual", "model.language_model.embed_tokens"]
post_transformer_module_names = ["model.language_model.norm", "lm_head"]
skip_layer_names = [
    "lm_head",
    "model.visual.*",               # 跳过视觉编码器
    "model.language_model.embed_tokens",  # 跳过 embedding
    "model.language_model.layers.*.mlp.gate",  # 跳过 MoE gate
]
```

**自定义加载器候选**：
```python
def get_model_loader_candidates(self):
    model_type = getattr(self.config, "model_type", None)
    candidates = []
    if model_type == "qwen3_vl_moe":
        candidates.append("Qwen3VLMoeForConditionalGeneration")
    elif model_type == "qwen3_vl":
        candidates.append("Qwen3VLForConditionalGeneration")
    candidates.append("AutoModelForImageTextToText")
    return candidates

def get_processor_loader_candidates(self):
    return ["AutoProcessor"]
```

**配置示例**：
```yaml
resources:
  - id: qwen3_vl
    type: Qwen3VLModel
    path: Qwen/Qwen3-VL-30B-A3B-Thinking
    model_hub: ms
    device_map: cuda
    trust_remote_code: true
    low_cpu_mem_usage: true
```

**注意事项**：
- VLM 模型会自动加载 `AutoProcessor`，校准数据集的 processor 参数由框架自动注入
- 仅语言模型部分参与量化，视觉编码器权重原样保留

---

## 3 接入新模型

### 3.1 完整步骤

1. **创建模型文件**：在 `src/npuslim/models/` 下创建以模型名命名的目录和模块文件
2. **继承 BaseLLMModel**：实现子类，配置层结构属性
3. **注册到 ModelRegistry**：使用 `@ModelRegistry.register()` 装饰器
4. **确保目录的 `__init__.py` 导出模块**

### 3.2 示例：接入 LLaMA 模型

**第一步**：创建目录结构

```
src/npuslim/models/llama/
├── __init__.py
└── llama_model.py
```

**第二步**：实现模型类

```python
# src/npuslim/models/llama/llama_model.py
from ..base_model import BaseLLMModel
from npuslim.core import ModelRegistry


@ModelRegistry.register("LLaMA", aliases=["LlamaModel", "Llama"])
class LlamaSlimModel(BaseLLMModel):
    """LLaMA model support for quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.block_name = "model.layers"
        self.post_transformer_module_names = ["model.norm", "lm_head"]
```

**第三步**：在 `__init__.py` 中导入

```python
# src/npuslim/models/llama/__init__.py
from .llama_model import LlamaSlimModel
```

**第四步**：使用配置

```yaml
resources:
  - id: llama
    type: LLaMA
    path: meta-llama/Llama-2-7b-hf
    device_map: cuda
    trust_remote_code: true
```

### 3.3 覆写指南

| 方法/属性 | 何时需要覆写 |
|-----------|-------------|
| `block_name` | Transformer 层命名前缀不同时（如 OPT 的 `model.decoder.layers`） |
| `pre_transformer_module_names` | embedding 模块命名不同或数量不同时 |
| `post_transformer_module_names` | 输出模块（norm、lm_head）命名不同时 |
| `skip_layer_names` | 需要跳过 MoE gate 或特殊模块时 |
| `get_model_loader_candidates()` | 非 CausalLM 架构（如 VL 模型）需要指定专用类时 |
| `get_processor_loader_candidates()` | VLM 模型需要加载 processor 时 |
| `model_type` | 设置为 `"VLM"` 以启用多模态处理流程 |

---

## 4 模型与算法兼容矩阵

| 模型 | INT8Dynamic | GPTQ | QuIP | SparseGPT |
|------|:-----------:|:----:|:----:|:---------:|
| Qwen3 (Dense) | Y | Y | Y | Y |
| Qwen3 (MoE) | Y | Y | Y | Y |
| OPT | Y | Y | Y | Y |
| GLM-5 (GlmMoeDsa) | Y | - | - | - |
| Qwen3-VL | Y | - | - | - |

**说明**：

- `Y` 表示已验证通过；`-` 表示理论可行但尚未提供官方配置
- 所有模型的 MoE gate 模块和 embedding/lm_head 默认不参与量化
- INT8Dynamic 适用于所有模型，因为其基于 PyTorch 原生量化 API
- GPTQ 需要 Hessian 统计，对 MoE 模型的混合专家层需要更长校准时间
- QuIP 和 SparseGPT 目前主要在 Qwen3 和 OPT 上经过验证

---

## 5 注册机制

NPUSlim 使用 `Registry` 单例模式管理所有模型类：

```python
# 注册模型
@ModelRegistry.register("MyModel", aliases=["MyModelAlias"])
class MyModel(BaseLLMModel):
    ...

# 通过注册名创建实例
model = ModelRegistry.create("MyModel", path="/path/to/model", device_map="cuda")
```

注册器支持懒加载：当 `models/__init__.py` 中使用 `register_lazy()` 延迟注册时，模块仅在实际访问时才导入，避免启动时加载所有模型依赖。
