# 流式分块管线设计文档

## 1. 设计动机

### 1.1 传统全量加载方案的瓶颈

大语言模型（LLM）的参数量增长迅速，从数十亿到数千亿参数不等。以一个 70B 参数模型为例，在 FP16 精度下仅权重就占用约 140 GiB 显存。传统量化管线需要将完整模型加载到内存/显存后才能执行算法处理，这带来了以下核心问题：

1. **峰值内存 = 模型权重 + 算法状态 + 输出缓冲**，总内存需求远超模型本身大小
2. **硬件门槛高**，必须有足够显存容纳整个模型，普通 GPU 无法处理超大模型
3. **无法水平扩展**，单卡显存成为硬上限

### 1.2 流式分块的核心思想

NPUSlim v2 采用**流式分块（Streaming Chunked）**架构，核心原则为：

- **分块加载**：每次仅加载 `chunk_size` 层 Transformer 层到内存
- **即时处理**：对当前 chunk 调用算法的 `process_chunk()` 进行变换
- **增量写入**：处理完毕后立即写入磁盘，释放内存
- **固定峰值**：内存占用与 `chunk_size` 成正比，与模型总层数无关

这使得在单张 GPU 上量化任意大小的模型成为可能——只需将 `chunk_size` 调整为显存允许的范围。

---

## 2. 整体数据流

从用户执行 `python tools/run.py -c config.yaml` 开始，到最终输出量化模型，完整管线流程如下：

```
YAML 配置文件
     |
     v
ConfigParser 解析 --> EngineConfig (metadata + resources + recipe)
     |
     v
SlimEngine 初始化
  ├── 创建 ResourceManager（持有全部 ResourceConfig）
  └── 遍历 recipe 中的每个 RecipeTaskConfig
       └── TaskRegistry.create() --> 创建 Task 实例（注入 resource_manager）
            |
            v
SlimEngine.run()
  └── 遍历 pipeline 中的 Task，依次调用 task.execute()
       |
       v
CompressorTask.execute()
  ├── on_start()
  |    ├── _create_model()       --> ResourceManager.acquire_model("@qwen3")
  |    ├── _create_data()        --> ResourceManager.acquire_dataset("@calib_data")
  |    ├── _create_algorithm()   --> AlgorithmRegistry.get("INT8Dynamic")()
  |    └── _create_saver()       --> SaverRegistry.create("StreamingHuggingFaceSaver")
  |
  └── run()
       ├── ChunkLoader.refresh_index()     --> 解析 safetensors 索引
       ├── algorithm.on_start()
       |
       ├── [循环] chunk_idx in [0, chunk_count):
       |    ├── loader.load_chunk(chunk_idx)   --> ChunkContext
       |    ├── chunk.calib_data = calib_data
       |    ├── algo.process_chunk(chunk)      --> 变换后的 ChunkContext
       |    ├── saver.add_tensors(chunk.all_tensors())
       |    └── loader.unload_chunk(chunk_idx)  --> 释放 GPU 内存
       |
       ├── [回填] 未被处理的原始张量
       ├── algo.on_finish()
       └── saver.finalize()                    --> 写入 index + 辅助文件
```

### 关键设计约束

- **SlimEngine 不持有模型对象**：仅持有 `ResourceManager`，由 Task 在运行时按需获取
- **Task 是执行单元**：所有资源获取（model、dataset、algorithm、saver）都在 Task 内部延迟完成
- **ChunkLoader 是无状态的迭代器**：`__iter__` 方法自动完成 load/unload 循环

---

## 3. 分块机制详解

### 3.1 ChunkLoader 的索引构建

`ChunkLoader` 支持四种检查点格式的自动检测，优先级从高到低：

| 优先级 | 格式 | 文件 |
|--------|------|------|
| 1 | Safetensors 分片索引 | `model.safetensors.index.json` |
| 2 | Safetensors 单分片 | `model.safetensors` |
| 3 | PyTorch Bin 分片索引 | `pytorch_model.bin.index.json` |
| 4 | PyTorch Bin 单分片 | `pytorch_model.bin` |

`refresh_index()` 方法在初始化阶段被调用一次，完成以下工作：

1. **解析索引文件**：读取 `weight_map`（`tensor_name -> shard_name` 的映射）
2. **构建分层映射**：通过正则 `^{block_name}\.(\d+)\.` 提取层索引，构建 `_layer_tensor_map`
3. **分类辅助模块**：`pre_module_names`（如 `model.embed_tokens`）和 `post_module_names`（如 `lm_head`）的张量分别归类
4. **检测未分配张量**：不属于任何已知分区的张量被标记为 `unassigned`，后续由 CompressorTask 回填

```python
# 索引构建后的内部数据结构
self._weight_map: Dict[str, str]            # "model.layers.0.self_proj.weight" -> "model-00001.safetensors"
self._tensor_names: List[str]               # 所有权重名称的有序列表
self._layer_tensor_map: Dict[int, List[str]] # 0 -> ["model.layers.0.attn.q weight", ...]
self._layer_indices: List[int]              # [0, 1, 2, ..., 31]
```

### 3.2 分块加载流程

`load_chunk(chunk_index)` 的执行逻辑：

```python
def load_chunk(self, chunk_index: int) -> ChunkContext:
    start = chunk_index * self.chunk_size
    end = min(start + self.chunk_size, self.get_total_layers())
    layer_indices = self._layer_indices[start:end]

    # 仅在第一个 chunk 加载 pre_modules（如 embed_tokens）
    pre_modules = self._load_module_infos(self._pre_module_tensor_map) if is_first_chunk else []

    # 加载当前 chunk 对应的 transformer 层
    layers = self._load_layers(layer_indices)

    # 仅在最后一个 chunk 加载 post_modules（如 lm_head）
    post_modules = self._load_module_infos(self._post_module_tensor_map) if is_last_chunk else []

    return ChunkContext(chunk_index=chunk_index, layers=layers,
                        pre_modules=pre_modules, post_modules=post_modules)
```

**关键细节**：

- 每个 chunk 包含 `chunk_size` 层 Transformer 层的全部权重
- `pre_modules` 仅在 `chunk_index == 0` 时加载
- `post_modules` 仅在最后一个 chunk 时加载
- 张量加载时使用 `safe_open()` 按需从磁盘读取，惰性解码到指定设备
- 分片句柄（shard handle）在 `_opened_shards` 中缓存，避免重复打开文件

### 3.3 ChunkContext 数据结构

`ChunkContext` 是管线中数据流转的核心载体：

```python
@dataclass
class ChunkContext:
    chunk_index: int                           # 当前 chunk 在全局中的序号
    layers: List[LayerInfo]                    # Transformer 层列表
    pre_modules: List[ModuleInfo]              # 前置模块（仅第一个 chunk）
    post_modules: List[ModuleInfo]             # 后置模块（仅最后一个 chunk）
    calib_data: Optional[Any] = None           # 校准数据 DataLoader
    metadata: Dict[str, Any] = {}              # 元数据（skip_layer_names, tensor_types 等）
```

其中 `LayerInfo` 和 `ModuleInfo` 是轻量级容器：

```python
@dataclass
class LayerInfo:
    name: str                        # "model.layers.12"
    index: int                       # 12
    tensors: Dict[str, torch.Tensor] # {"self_attn.q_proj.weight": tensor, ...}

@dataclass
class ModuleInfo:
    name: str                        # "model.embed_tokens"
    tensors: Dict[str, torch.Tensor] # {"weight": tensor}
```

`ChunkContext` 提供了多种张量访问方式：

- `all_tensors()` → 返回全限定名到张量的扁平映射
- `get_tensor(name)` / `update_tensor(name, tensor)` → 按全限定名读写
- `filter_by_prefix(prefix)` → 按前缀筛选张量子集
- `tensor_count` / `tensor_names` → 便捷属性

**设计原则**：`LayerInfo.tensors` 中的 key 是**相对于层名**的短名（如 `self_attn.q_proj.weight`），而 `all_tensors()` 返回的是全限定名。这种设计使得算法处理时无需关心全局命名，而在保存时能还原完整路径。

---

## 4. 算法处理接口

### 4.1 BaseAlgorithm 生命周期

```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def process_chunk(self, chunk: ChunkContext) -> ChunkContext:
        """处理一个 chunk，返回变换后的 ChunkContext（可原地修改）"""
        raise NotImplementedError

    def on_start(self) -> None:
        """所有 chunk 处理之前调用，用于初始化内部状态"""
        pass

    def on_finish(self) -> None:
        """所有 chunk 处理完毕后调用，用于清理资源"""
        pass
```

调用时序保证：

```
algo.on_start()
  ├── algo.process_chunk(chunk_0)
  ├── algo.process_chunk(chunk_1)
  ├── ...
  └── algo.process_chunk(chunk_N-1)
algo.on_finish()
```

### 4.2 BaseQuantizationAlgorithm 扩展

量化算法通常继承 `BaseQuantizationAlgorithm`，获得以下能力：

- **`set_runtime_context()`**：接收 `model_obj`、`model_config`、`skip_layer_names`，在第一个 chunk 之前调用
- **`should_skip_name()`**：根据 glob/regex 模式判断某层是否应跳过量化
- **`target_backend` 属性**：检测当前运行后端（CPU/CUDA/NPU），用于选择输出格式

```python
class BaseQuantizationAlgorithm(BaseAlgorithm):
    def set_runtime_context(self, *, model_obj, model_config, skip_layer_names):
        self._model_obj = model_obj
        self._model_config = model_config
        self._skip_layer_names = list(skip_layer_names)

    @staticmethod
    def should_skip_name(full_name: str, skip_layer_names: Iterable[str]) -> bool:
        for skip_name in skip_layer_names:
            if skip_name.startswith("re:"):
                if re.fullmatch(skip_name[3:], full_name):
                    return True
            elif full_name == skip_name or full_name.startswith(f"{skip_name}."):
                return True
            elif fnmatch.fnmatch(full_name, skip_name):
                return True
        return False
```

### 4.3 典型算法实现模式

一个简化版的 INT8 动态量化算法：

```python
@AlgorithmRegistry.register("INT8Dynamic")
class INT8DynamicAlgorithm(BaseQuantizationAlgorithm):
    _TAG = "INT8"

    def on_start(self):
        self._quantized_count = 0

    def process_chunk(self, chunk: ChunkContext) -> ChunkContext:
        skip_names = self._set_skip_from_chunk_metadata(chunk)

        for layer in chunk.layers:
            for tensor_name, tensor in layer.tensors.items():
                full_name = f"{layer.name}.{tensor_name}"
                if self.should_skip_name(full_name, skip_names):
                    continue
                if tensor.ndim == 2:  # 仅对 2D 权重做量化
                    layer.tensors[tensor_name] = self._quantize_weight(tensor)
                    self._quantized_count += 1
        return chunk

    def on_finish(self):
        logger.info(f"INT8 quantized {self._quantized_count} tensors")
```

---

## 5. 流式保存

### 5.1 StreamingHuggingFaceSaver 设计

`StreamingHuggingFaceSaver` 实现了**基于大小阈值的自动分片写入**：

核心写入流程：

```
add_tensor(name, tensor)
  ├── 计算 tensor_size = tensor.numel() * tensor.element_size()
  ├── 如果 buffer_size + tensor_size > size_threshold 且 buffer 非空:
  |    └── flush() --> 将 buffer 写入 safetensors 分片文件
  ├── tensor 移至 CPU，加入 buffer
  └── 如果 buffer_size >= size_threshold:
       └── flush() --> 立即写入
```

`flush()` 的具体操作：

1. 生成分片文件名（如 `model-00001.safetensors`）
2. 调用 `safetensors.torch.save_file(buffer, shard_path)` 原子写入
3. 更新 `weight_map`，记录每个张量所在的分片文件
4. 清空 buffer，递增 shard_counter

### 5.2 finalize() 完整流程

当所有 chunk 处理完毕后，`CompressorTask` 调用 `saver.finalize()`：

1. 刷出残余 buffer
2. 写入 `model.safetensors.index.json`（weight_map + metadata）
3. 保存 HF 资产（config.json, tokenizer, processor）
4. 从源模型复制辅助文件（README, generation_config.json 等）
5. [NPU] 生成 `quant_model_description.json`

---

## 6. 内存管理

### 6.1 固定内存占用的实现

整个管线的内存占用分析：

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| ChunkLoader 分片索引 | O(1) MiB | 仅存储 weight_map 字典，无张量 |
| 当前 chunk 张量 | chunk_size * 层参数量 | 与 chunk_size 线性相关 |
| 算法内部状态 | 取决于算法 | GPTQ 需 Hessian 矩阵，INT8 几乎无额外开销 |
| Saver buffer | <= size_threshold | 写入磁盘后立即释放 |
| 模型元数据 | O(1) MiB | config + tokenizer + empty_model（meta device） |

**峰值内存 ≈ chunk 张量 + 算法状态 + saver 阈值**

对于 `chunk_size=4` 的 70B 模型量化场景：
- 每个 Transformer 层约 4 GiB（FP16），4 层 ≈ 16 GiB
- Saver buffer 最大 4 GiB
- 峰值约 20 GiB，远低于全量加载的 140 GiB

### 6.2 资源释放机制

```python
# CompressorTask.run() 中的资源释放链
try:
    for chunk_idx in range(chunk_count):
        chunk = loader.load_chunk(chunk_idx)    # 加载 chunk
        chunk = algo.process_chunk(chunk)       # 算法处理
        saver.add_tensors(chunk.all_tensors())   # 增量写入
        loader.unload_chunk(chunk_idx)           # 关闭分片句柄 + 清空 GPU 缓存
finally:
    algo.on_finish()                             # 算法资源释放
    saver.finalize()                             # 最终刷盘
    loader.close()                               # 清空所有索引和句柄
```

### 6.3 未处理张量的回填

某些张量（如 `model.embed_tokens.weight`、`lm_head.weight`）可能未被算法处理（被 skip 规则跳过）。CompressorTask 在主循环结束后执行回填：检测所有原始张量中被算法"遗漏"的部分，重新加载并传递给 saver。这确保了输出模型是完整的，即使部分张量未经过量化处理。

---

## 7. 总结

NPUSlim v2 的流式分块管线通过以下设计实现了"固定内存占用"的目标：

1. **ChunkLoader**：按需加载 `chunk_size` 层张量，惰性解码到目标设备
2. **ChunkContext**：轻量级数据容器，算法通过 `process_chunk()` 接口对其进行变换
3. **StreamingHuggingFaceSaver**：基于大小阈值的增量写入，buffer 满即刷盘
4. **BaseAlgorithm** 生命周期：`on_start` -> `process_chunk` * N -> `on_finish`
5. **严格资源释放**：每个 chunk 处理完毕后立即卸载，`finally` 块保证清理

该设计使得在 16 GiB 显存的 GPU 上量化 70B+ 参数模型成为可能，只需设置合适的 `chunk_size`。
