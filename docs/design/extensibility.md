# 可扩展性设计文档

## 1. 概述

NPUSlim v2 采用**注册表驱动的插件式架构**。框架的核心流转逻辑（配置解析、资源管理、分块加载、流式保存）与具体实现（算法、模型、数据集、保存器）完全解耦。用户通过实现基类并注册到对应的全局注册表，即可无缝扩展框架能力。

本框架提供五类可扩展点：

| 扩展点 | 注册表 | 基类 | 典型场景 |
|--------|--------|------|----------|
| 算法 | `AlgorithmRegistry` | `BaseAlgorithm` | 新增量化/剪枝算法 |
| 模型 | `ModelRegistry` | `BaseLLMModel` | 适配新的 LLM 架构 |
| 数据集 | `DatasetRegistry` | `BaseDataset` | 添加新的校准数据源 |
| 任务 | `TaskRegistry` | `BaseTask` | 定义新的管线任务类型 |
| 保存器 | `SaverRegistry` | `BaseSaver` | 支持新的输出格式 |

---

## 2. Registry 注册表模式

### 2.1 Registry 类设计

`Registry` 是一个通用的注册表实现，支持**即时注册**和**懒加载注册**两种模式：

```python
class Registry:
    def __init__(self, name: str, submodule: Optional[str] = None):
        self.name = name
        self._submodule = submodule          # 关联的子模块路径
        self._registry: Dict[str, Type] = {} # name -> class 的即时注册表
        self._aliases: Dict[str, str] = {}   # alias -> primary_name
        self._lazy_map: Dict[str, str] = {}  # name -> module_path 的懒加载映射
        self._submodule_loaded = False        # 子模块是否已导入
```

### 2.2 五个全局单例注册表

框架在 `core/factory.py` 中定义了五个全局单例：

```python
AlgorithmRegistry = Registry("Algorithm", "algorithms")
ModelRegistry     = Registry("Model", "models")
DatasetRegistry   = Registry("Dataset", "datasets")
TaskRegistry      = Registry("Task", "tasks")
SaverRegistry     = Registry("Saver", "savers")
```

### 2.3 核心 API

#### register() —— 即时注册

使用装饰器将类立即注册到注册表：

```python
@AlgorithmRegistry.register("MyAlgo", aliases=["my_algo", "MyAlgorithm"])
class MyAlgorithm(BaseAlgorithm):
    def process_chunk(self, chunk):
        return chunk
```

#### register_lazy() —— 懒加载注册

声明注册名到模块路径的映射，实际 import 延迟到首次 `get()` 调用时：

```python
AlgorithmRegistry.register_lazy(
    "INT8Dynamic",
    ".quantization.int8_dynamic.int8_dynamic_algo",
    aliases=["INT8Dyn", "int8_dyn"],
)
```

#### get() —— 按名称获取类

```python
algo_cls = AlgorithmRegistry.get("INT8Dynamic")   # 通过主名
algo_cls = AlgorithmRegistry.get("int8_dyn")       # 通过别名
```

#### create() —— 按名称创建实例

```python
instance = AlgorithmRegistry.create("INT8Dynamic", wbits=8)
```

#### list() —— 列出所有已注册名称

```python
names = AlgorithmRegistry.list()
```

---

## 3. 懒加载机制详解

### 3.1 为什么需要懒加载

大模型工具链涉及大量重型依赖（torch, transformers, safetensors 等）。如果启动时导入所有算法模块，即使最终只使用一个算法，也会付出全量导入的时间成本。懒加载机制保证：**只有被实际使用的算法/模型/数据集才会被导入**。

### 3.2 懒加载的触发链路

以配置 `algorithm: {type: GPTQ}` 为例：

```
CompressorTask._create_algorithm()
  └── AlgorithmRegistry.get("GPTQ")
       ├── 检查 _registry["gptq"] → 未命中
       ├── _ensure_submodule_loaded()
       |    └── importlib.import_module("npuslim.algorithms")
       |         └── algorithms/__init__.py 执行 register_lazy()
       ├── 检查 _lazy_map["gptq"] → 命中
       ├── importlib.import_module("npuslim.algorithms.quantization.gptq.gptq_algo")
       |    └── @AlgorithmRegistry.register("GPTQ") 装饰器执行
       └── 返回 _registry["gptq"]
```

### 3.3 最佳实践

在子模块的 `__init__.py` 中使用 `register_lazy()` 声明所有可用的实现，在具体实现文件中使用 `@register()` 装饰器完成真正的注册。这种**两级注册**确保了：
- `list()` 能列出所有可用实现（无需全部导入）
- `get()` 只导入需要的模块

---

## 4. @resource 引用机制

### 4.1 配置中的资源声明与引用

YAML 配置使用 `resources` 列表声明可用资源，`recipe` 中通过 `@id` 语法引用：

```yaml
resources:
  - id: qwen3
    type: Qwen3
    path: Qwen/Qwen3-0.6B
    device_map: cuda

  - id: calib_data
    type: C4
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"           # 引用 id=qwen3 的资源
    dataloader:
      dataset: "@calib_data"  # 引用 id=calib_data 的资源
```

### 4.2 ResourceManager 的解析逻辑

`ResourceManager` 持有全部 `ResourceConfig`，在 Task 请求资源时执行延迟实例化。同一资源在多个 Task 间共享实例（缓存机制），避免重复加载。

### 4.3 资源配置的 extra 字段

`ResourceConfig.extra` 是一个字典，包含除 `id` 和 `type` 之外的所有配置字段，作为关键字参数传递给构造函数：

```python
# YAML 中:
#   path: Qwen/Qwen3-0.6B
#   device_map: cuda
# 等效于:
ModelRegistry.create("Qwen3", path="Qwen/Qwen3-0.6B", device_map="cuda")
```

---

## 5. 扩展点总结与完整示例

### 5.1 扩展新算法

**步骤 1**：创建算法实现文件。

```python
# src/npuslim/algorithms/quantization/w4a16/w4a16_algo.py
import torch
from npuslim.algorithms.quantization.base_quant_algo import BaseQuantizationAlgorithm
from npuslim.core import AlgorithmRegistry


@AlgorithmRegistry.register("W4A16", aliases=["w4a16", "W4A16Quant"])
class W4A16Algorithm(BaseQuantizationAlgorithm):
    _TAG = "W4A16"

    def __init__(self, group_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def on_start(self):
        self._quantized_count = 0

    def process_chunk(self, chunk):
        skip_names = self._set_skip_from_chunk_metadata(chunk)

        for layer in chunk.layers:
            for tensor_name, tensor in list(layer.tensors.items()):
                full_name = f"{layer.name}.{tensor_name}"
                if self.should_skip_name(full_name, skip_names):
                    continue
                if tensor.ndim == 2 and tensor.shape[0] % self.group_size == 0:
                    quantized, scale, zero_point = self._quantize_w4(tensor)
                    layer.tensors[tensor_name] = quantized
                    layer.tensors[f"{tensor_name}_scale"] = scale
                    layer.tensors[f"{tensor_name}_zero_point"] = zero_point
                    self._quantized_count += 1
        return chunk

    def _quantize_w4(self, tensor: torch.Tensor):
        # 将权重分组量化为 4-bit
        ...
        return quant_tensor, scale, zp

    def on_finish(self):
        print(f"W4A16 quantized {self._quantized_count} weight tensors")
```

**步骤 2**：在 `algorithms/__init__.py` 中声明懒加载。

```python
AlgorithmRegistry.register_lazy(
    "W4A16", ".quantization.w4a16.w4a16_algo", aliases=["w4a16", "W4A16Quant"])
```

**步骤 3**：在 YAML 配置中使用。

```yaml
recipe:
  - name: "W4A16_Quantization"
    type: compressor
    model: "@my_model"
    algorithm:
      type: W4A16
      group_size: 128
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: ./outputs/w4a16
```

### 5.2 扩展新模型

```python
# src/npuslim/models/llama4.py
from npuslim.models.base_model import BaseLLMModel
from npuslim.core import ModelRegistry


@ModelRegistry.register("LLaMA4", aliases=["Llama4", "llama4"])
class LLaMA4Model(BaseLLMModel):
    """LLaMA-4 model adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_name = "model.layers"
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.post_transformer_module_names = ["model.norm", "lm_head"]
        self.skip_layer_names = ["lm_head"]
```

### 5.3 扩展新数据集

```python
# src/npuslim/datasets/custom_json_dataset.py
import json
from pathlib import Path
from npuslim.datasets.base_dataset import BaseDataset
from npuslim.core import DatasetRegistry


@DatasetRegistry.register("CustomJSON", aliases=["custom_json"])
class CustomJSONDataset(BaseDataset):
    """从 JSON Lines 文件加载校准数据。"""

    def __init__(self, *, data_dir: str, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.split = split
        self._load_data()

    def _load_data(self):
        files = sorted(Path(self.data_dir).glob(f"{self.split}_*.json"))
        count = 0
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if count >= self.num_samples:
                        break
                    item = json.loads(line.strip())
                    text = item.get("text", "")
                    enc = self.processor(text, return_tensors="pt",
                                         max_length=self.max_seq_length, truncation=True)
                    if enc["input_ids"].shape[1] == 0:
                        continue
                    self.data.append({
                        "input_ids": enc["input_ids"].to(self.device),
                        "attention_mask": enc["attention_mask"].to(self.device),
                        "labels": enc["input_ids"].roll(-1, dims=-1).to(self.device),
                    })
                    count += 1
```

### 5.4 扩展新保存器

```python
# src/npuslim/savers/gguf_saver.py
from npuslim.savers.base_saver import BaseSaver
from npuslim.core import SaverRegistry


@SaverRegistry.register("GGUFSaver", aliases=["gguf"])
class GGUFSaver(BaseSaver):
    """GGUF format saver (simplified example)."""

    def add_tensor(self, name, tensor, tensor_type=None):
        self.buffer[name] = tensor.cpu()

    def flush(self):
        return None

    def finalize(self):
        output_path = self.output_dir / "model.gguf"
        self._write_gguf(output_path, self.buffer)
```

---

## 6. 完整扩展清单

新增一个扩展组件的标准化流程：

| 步骤 | 操作 | 位置 |
|------|------|------|
| 1 | 创建实现文件，继承对应基类，使用 `@Registry.register()` 装饰器 | `src/npuslim/{category}/...` |
| 2 | 在对应子模块的 `__init__.py` 中添加 `register_lazy()` 声明 | `src/npuslim/{category}/__init__.py` |
| 3 | 在 YAML 配置中使用新注册的 `type` 名称 | 用户配置文件 |

关键约束：
- **注册名不区分大小写**，内部统一小写存储
- **别名不能冲突**，跨注册表的别名不会冲突（因为是不同的 Registry 实例）
- **懒加载路径**推荐使用相对路径（以 `.` 开头）
- **构造函数参数**必须与 YAML 配置中的字段名一致（`extra` 字段透传）
- **实例缓存**在 `ResourceManager` 层面完成，同一 `@id` 只创建一次

---

## 7. 设计原则

1. **开放封闭原则**：新增扩展无需修改框架核心代码，只需注册和声明
2. **延迟加载**：所有实现类在首次使用时才导入，减少启动开销
3. **约定优于配置**：基类提供合理的默认行为，子类只需覆盖差异部分
4. **配置驱动**：通过 YAML 配置组合资源与算法，无需编写代码即可切换量化方案
