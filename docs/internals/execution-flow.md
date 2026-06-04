# NPUSlim v2 完整执行流程

本文档从 `python tools/run.py -c config.yaml` 命令出发，逐步描述整个量化流程中每个阶段的行为。

---

## 1 总体架构概览

NPUSlim v2 采用 **流式分块（streaming chunk-based）** 设计。核心思想是不将模型完整加载到内存，而是通过 `ChunkLoader` 逐 chunk 流式加载权重，经由算法处理后由 `StreamingHuggingFaceSaver` 增量写入磁盘。

执行流程分为四个阶段：

```
启动阶段 → 引擎初始化 → 任务执行 → 结果输出
```

---

## 2 启动阶段：bootstrap_from_path()

入口文件 `tools/run.py` 调用 `bootstrap_from_path(cfg_path)` 完成运行时初始化。该函数定义在 `src/npuslim/core/bootstrap.py`。

### 2.1 YAML 加载

```python
raw_cfg = _load_raw_yaml(cfg_path)
```

使用 `yaml.safe_load()` 读取 YAML 文件，校验顶层结构必须为字典。

### 2.2 日志配置

```python
resolved_log_dir = resolve_log_dir(cfg_path, raw_cfg, override=log_dir)
setup_logger(log_dir=resolved_log_dir)
dump_config_snapshot(raw_cfg, resolved_log_dir)
```

- 根据配置文件路径自动推导日志目录
- 配置 loguru 日志输出
- 将原始 YAML 配置快照保存到日志目录，便于回溯

### 2.3 配置解析

```python
parsed_cfg = parse_config(cfg_path)
```

`parse_config()`（`src/npuslim/config/parser.py`）将原始 YAML 字典解析为结构化的 `EngineConfig` 对象：

```
EngineConfig
├── metadata: MetadataConfig        # name, description
├── resources: List[ResourceConfig]  # id, type, extra(所有其他参数)
└── recipe: List[RecipeTaskConfig]   # name, type, model, dataloader, algorithm, saver, extra
```

`ResourceConfig` 将 YAML 中的 `path`、`model_hub`、`device_map` 等参数全部收入 `extra` 字典，交由具体模型/数据集类的构造函数解析。

### 2.4 输出路径策略

```python
apply_saver_path_policy(parsed_cfg, cfg_path)
```

根据配置文件的相对路径自动推导 saver 的 `save_path`。例如配置文件为 `configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml`，saver 的 `save_dir` 为 `./outputs`，则最终 `save_path` 解析为 `./outputs/qwen3/int8_dynamic/qwen3_8b-w8a8`。

### 2.5 配置校验

```python
validate_config(parsed_cfg, strict=strict_validate)
```

校验内容包括：
- 资源引用（`@id` 格式）是否存在对应的 resource 声明
- 必填字段是否完整
- 类型值是否在注册表中存在

### 2.6 配置展示

```python
print_config(parsed_cfg, title=f"Configuration of {cfg_path}")
show_npuslim_header()
```

格式化打印解析后的配置信息，并显示 NPUSlim 横幅。

---

## 3 引擎初始化：SlimEngine

`bootstrap_from_path()` 返回 `EngineConfig`，随后由调用方传入 `SlimEngine`：

```python
engine = SlimEngine(config=parsed_cfg)
engine.run()
```

### 3.1 ResourceManager 创建

```python
self.rm = ResourceManager(resources=self.config.resources)
```

`ResourceManager`（`src/npuslim/core/resource_manager.py`）以惰性方式管理模型和数据集资源：

- 将 `ResourceConfig` 列表建立 `id → config` 索引
- 仅在首次访问时实例化具体对象（懒加载）
- 通过 `_resolve_resource_kind()` 自动判断资源类型（模型 or 数据集），查询 `ModelRegistry` 和 `DatasetRegistry`

### 3.2 构建执行管线

```python
def _build_pipeline(self):
    for task_config in self.config.recipe:
        task_kwargs = task_config.extra
        task = TaskRegistry.create(
            type_name=task_config.type,
            name=task_config.name,
            model=task_config.model,       # "@qwen3"
            dataloader=task_config.dataloader,
            algorithm=task_config.algorithm,
            saver=task_config.saver,
            resource_manager=self.rm,
            **task_kwargs,
        )
        self.pipeline.append(task)
```

遍历 `recipe` 列表，通过 `TaskRegistry` 创建任务实例。每个任务接收 `resource_manager` 引用，由任务自身决定何时获取哪些资源。这种设计使引擎保持最小化，任务具有完全的灵活性。

### 3.3 管线执行

```python
def run(self):
    for idx, task in enumerate(self.pipeline):
        result = task.execute()
        results.append(result)
    return results
```

顺序执行管线中的所有任务。当前版本中，管线通常只包含一个 `compressor` 类型的任务。

---

## 4 任务执行：BaseTask 生命周期

`BaseTask`（`src/npuslim/tasks/base_task.py`）定义了统一的生命周期框架：

```python
def execute(self):
    self.on_start()      # 阶段 1：初始化
    try:
        return self.run()  # 阶段 2：核心逻辑（子类实现）
    finally:
        self.on_finish()   # 阶段 3：收尾
```

### 4.1 on_start 阶段

`BaseTask.on_start()` 的默认实现完成四个组件的创建：

```python
def on_start(self):
    self._create_model()       # 1. 通过 ResourceManager 惰性获取模型对象
    self._create_data()        # 2. 创建校准数据 DataLoader
    self._algorithm = self._create_algorithm()  # 3. 创建算法实例
    self._saver = self._create_saver()          # 4. 创建 Saver 实例
```

**模型获取（`_create_model`）**：

```python
self._model_obj = self.rm.acquire_model(self.model_ref)
```

`ResourceManager.acquire_model()` 的工作流程：
1. 检查缓存：若已实例化则直接返回
2. 查找 ResourceConfig
3. 通过 `_resolve_resource_kind()` 确认类型为 model
4. 调用 `ModelRegistry.create(cfg.type, **cfg.extra)` 创建实例
5. 缓存实例并返回

模型对象在首次创建时调用 `prepare_metadata()`，加载 tokenizer、config 等轻量元数据。

**校准数据创建（`_create_data`）**：

```python
dataset = self.rm.acquire_dataset(self.data_ref, processor=processor)
self._calib_data = DataLoader(dataset, **loader_kwargs)
```

VLM 模型使用 `processor`，LLM 模型使用 `tokenizer` 作为 processor 参数。DataLoader 自动继承数据集类的 `collate_fn`。

**算法创建（`_create_algorithm`）**：

从 `algorithm` 配置字典中取出 `type` 字段，通过 `AlgorithmRegistry` 查找并实例化。其余字段作为关键字参数传入。

**Saver 创建（`_create_saver`）**：

默认创建 `StreamingHuggingFaceSaver`。

### 4.2 run 阶段

由子类实现核心逻辑。对于 `CompressorTask`，这是最核心的部分（详见第 5 节）。

### 4.3 on_finish 阶段

基类默认实现仅打印日志。子类可覆写以执行清理工作。

---

## 5 CompressorTask 核心流程

`CompressorTask`（`src/npuslim/tasks/compressor/task.py`）是 NPUSlim v2 的核心任务类，注册为 `"compressor"`。它实现了完整的流式量化管线。

### 5.1 初始化参数

```python
class CompressorTask(BaseTask):
    def __init__(self, *, execution=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = execution.get("mode", "full")      # "full" 或 "streaming"
        self.chunk_size = max(int(execution.get("chunk_size", 1)), 1)
```

- `mode="full"`：一次性加载所有张量
- `mode="streaming"`：按 `chunk_size` 分块加载，节省内存

### 5.2 run() 方法完整流程

#### 步骤 1：创建 ChunkLoader

```python
loader = self._create_loader()
```

`ChunkLoader` 根据 `block_name`、`pre_module_names`、`post_module_names` 构建张量索引。它会自动检测模型目录中的检查点格式：

1. `model.safetensors.index.json` — 分片 safetensors 索引（优先）
2. `model.safetensors` — 单文件 safetensors
3. `pytorch_model.bin.index.json` — 分片 PyTorch bin 索引
4. `pytorch_model.bin` — 单文件 PyTorch bin

`ChunkLoader` 同时支持本地路径和远程 Hub 模型（HuggingFace / ModelScope）。

#### 步骤 2：刷新张量索引

```python
loader.refresh_index()
```

解析检查点索引文件，构建以下映射：

- `_weight_map`：`{tensor_name: shard_name}` — 每个张量所属的分片
- `_layer_tensor_map`：`{layer_idx: [tensor_names]}` — 按层索引的张量
- `_pre_module_tensor_map` / `_post_module_tensor_map` — 前置/后置模块的张量
- `_unassigned_tensor_names` — 未归类的张量（会触发警告，最终由 backfill 机制补写）

#### 步骤 3：配置 Saver

```python
saver.set_source(loader.resolve_model_source(), ...)
saver.set_hf_assets(model_config=config, tokenizer=tokenizer, processor=processor)
```

告知 Saver 模型的源路径（用于复制辅助文件）以及需要保存的 HF 资产（config.json、tokenizer 文件等）。

#### 步骤 4：解析跳过层

```python
resolved_skip_layer_names = self._resolve_skip_layer_names(loader)
```

**skip-layer 机制** 的完整流程：

1. 合并模型内置的 `skip_layer_names` 和用户配置的 `ignore_layers`
2. 根据所有张量名构建候选匹配集（包含完整张量名和去掉最后一段的模块名）
3. 对每个模式进行匹配：
   - `"re:pattern"` 前缀：使用 `re.fullmatch()` 正则匹配
   - 精确匹配：直接查找
   - glob 匹配：使用 `fnmatch.filter()` 通配符匹配

例如，`"model.layers.*.mlp.gate"` 会匹配 `model.layers.0.mlp.gate`、`model.layers.1.mlp.gate` 等所有层的 gate 模块。

#### 步骤 5：注入运行时上下文

```python
algo.set_runtime_context(
    model_obj=self._model_obj,
    model_config=config,
    skip_layer_names=resolved_skip_layer_names,
)
```

将模型对象、配置、跳过层名注入算法实例。`BaseQuantizationAlgorithm` 会利用这些信息在 `process_chunk()` 中过滤不需要处理的层。

#### 步骤 6：分块处理循环

```python
algo.on_start() 
try:
    if self.mode == "full":
        chunk = loader.load_full()
        # ... 处理
    else:
        for chunk_idx in range(chunk_count):
            chunk = loader.load_chunk(chunk_idx)
            # ... 处理
            loader.unload_chunk(chunk_idx)
finally:
    algo.on_finish()
    saver.finalize()
    loader.close()
```

**每个 chunk 的处理流程**：

```
loader.load_chunk(chunk_idx)
       ↓
chunk.calib_data = self._calib_data      # 注入校准数据
chunk.metadata["skip_layer_names"] = ...  # 注入跳过层信息
       ↓
algo.process_chunk(chunk)                 # 算法处理（核心）
       ↓
saver.add_tensors(chunk.all_tensors())    # 增量写入
       ↓
loader.unload_chunk(chunk_idx)            # 释放内存
```

#### 步骤 7：Backfill 补写

```python
missing_original_keys = sorted(all_original_keys - touched_original_keys)
if missing_original_keys:
    missing_tensors = loader.load_tensors(missing_original_keys)
    saver.add_tensors(missing_tensors, tensor_types={"FLOAT": ...})
```

对于所有未被任何 chunk 覆盖的原始张量（如 `_unassigned_tensor_names` 中的张量），以原始精度补写到输出目录，确保输出模型完整。

#### 步骤 8：收尾与状态发布

```python
saver.finalize()  # 写入 index.json、复制 config/tokenizer 等
self.rm.publish_model_state(model_ref, model_obj, state_meta={...})
```

`StreamingHuggingFaceSaver.finalize()` 完成以下工作：
- 刷新所有缓冲张量到磁盘
- 生成 `model.safetensors.index.json`
- 复制 `config.json`、tokenizer 文件、processor 文件等
- NPU 模式下额外生成 `quant_model_description.json`

`publish_model_state()` 将量化后的模型状态更新到 ResourceManager，便于下游任务引用。

---

## 6 ChunkContext 结构

`ChunkContext`（`src/npuslim/tasks/compressor/context.py`）是 chunk 处理过程中的核心数据结构，承载了算法所需的所有信息。

### 6.1 数据结构定义

```python
@dataclass
class LayerInfo:
    name: str                              # 完整层名，如 "model.layers.12"
    index: int                             # 全局层索引
    tensors: Dict[str, torch.Tensor]       # 相对于层的张量名 → 张量

@dataclass
class ModuleInfo:
    name: str                              # 模块名，如 "model.embed_tokens"
    tensors: Dict[str, torch.Tensor]       # 相对于模块的张量名 → 张量

@dataclass
class ChunkContext:
    chunk_index: int                       # chunk 序号
    layers: List[LayerInfo]                # Transformer 层列表
    pre_modules: List[ModuleInfo]          # Transformer 前置模块
    post_modules: List[ModuleInfo]         # Transformer 后置模块
    calib_data: Optional[Any]              # 校准数据 DataLoader
    metadata: Dict[str, Any]               # 元数据（skip_layer_names、tensor_types 等）
```

### 6.2 ChunkContext 关键方法

| 方法 | 说明 |
|------|------|
| `all_tensors()` | 返回 `{完整张量名: Tensor}` 的扁平字典 |
| `get_tensor(name)` | 按完整名获取张量 |
| `update_tensor(name, tensor)` | 更新/添加张量（自动定位到正确的层/模块） |
| `filter_tensors(patterns)` | 按正则模式筛选张量 |
| `filter_by_prefix(prefix)` | 按前缀筛选张量 |
| `is_first_chunk` | 是否为第一个 chunk（属性） |
| `layer_indices` | chunk 内所有层的全局索引（属性） |
| `tensor_count` | chunk 中张量总数（属性） |

### 6.3 ChunkContext 传递给算法的信息

算法的 `process_chunk(chunk)` 接收的 chunk 包含：

1. **权重张量**：通过 `layers`、`pre_modules`、`post_modules` 访问
2. **校准数据**：通过 `chunk.calib_data` 获取 DataLoader，用于 GPTQ 的 Hessian 收集或 INT8 的校准统计
3. **跳过层信息**：`chunk.metadata["skip_layer_names"]` 标识哪些层不参与量化
4. **张量类型标注**：`chunk.metadata["tensor_types"]` 在算法处理后写入，标识每个张量的量化状态（NPU 模式必须）

---

## 7 ChunkLoader 流式加载机制

`ChunkLoader`（`src/npuslim/tasks/compressor/loader.py`）负责按 chunk 加载模型权重。

### 7.1 索引构建

`refresh_index()` 解析检查点索引文件，按 `block_name` 的正则模式将张量分配到各层：

```python
# block_name = "model.layers" 时
# 匹配 model.layers.0.xxx、model.layers.1.xxx 等
pattern = re.compile(rf"^model\.layers\.(\d+)\.")
```

未匹配到任何层的张量归入 `_unassigned_tensor_names`，由 `pre_module_names` 和 `post_module_names` 进一步分配。

### 7.2 分片缓存

`ChunkLoader` 维护一个打开的分片文件缓存：

```python
self._opened_shards: Dict[str, Any] = {}
```

对于 safetensors 格式，使用 `safe_open()` 延迟读取；对于 torch_bin 格式，使用 `torch.load()` 整体加载。分片在 `unload_chunk()` 时关闭，释放内存。

### 7.3 chunk 划分策略

```python
def get_chunk_count(self):
    total_layers = len(self._layer_indices)
    return (total_layers + self.chunk_size - 1) // self.chunk_size
```

- `load_chunk(chunk_idx)`：加载 `[start, end)` 范围内的层
- 首个 chunk 额外加载 `pre_modules`（embedding 等）
- 末尾 chunk 额外加载 `post_modules`（norm、lm_head 等）
- `load_full()` 等价于加载所有层 + pre + post

---

## 8 算法处理：process_chunk()

### 8.1 BaseQuantizationAlgorithm 基类

所有量化算法继承自 `BaseQuantizationAlgorithm`，它提供：

- `set_runtime_context()`：注入模型对象、配置、跳过层
- `should_skip_name()`：判断一个张量是否应跳过，支持精确匹配、前缀匹配、glob 通配符、正则表达式
- `_mark_model_quantized()`：标记模型为已量化

### 8.2 生命周期钩子

```python
class BaseAlgorithm(ABC):
    def on_start(self) -> None:     # 全局初始化（如加载校准模型）
    def process_chunk(self, chunk) -> ChunkContext:  # 处理单个 chunk
    def on_finish(self) -> None:    # 全局收尾
```

`on_start` 在所有 chunk 处理之前调用一次。例如 GPTQ 在 `on_start` 中构建空壳模型并运行校准推理收集 Hessian 矩阵。`on_finish` 在所有 chunk 处理完成后调用一次。

### 8.3 算法处理流程（以 GPTQ 为例）

```
on_start():
  → prepare_empty_model()           # meta device 骨架
  → 运行校准推理 → 收集 Hessian 统计

for each chunk:
  process_chunk(chunk):
    → 遍历 chunk 中的每个 LayerInfo
    → 对每个层中的 Linear 层权重：
       - 检查 should_skip_name() → 跳过或量化
       - 使用 Hessian 信息计算最优量化权重
       - 更新 chunk 中的张量
    → 标记 chunk.metadata["tensor_types"]
    → 返回修改后的 chunk

on_finish():
  → 释放 Hessian 资源
  → _mark_model_quantized()
```

### 8.4 INT8Dynamic 处理流程

INT8 动态量化不需要 Hessian 收集，流程更为简洁：

```
on_start():
  → 初始化量化参数（wbits、量化方法等）

for each chunk:
  process_chunk(chunk):
    → 遍历 chunk 中的每个层
    → 对权重进行 per-channel 量化
    → 生成量化参数（scale、zero_point）
    → 更新 chunk 中的张量
    → 标记 chunk.metadata["tensor_types"]

on_finish():
  → _mark_model_quantized()
```

---

## 9 StreamingHuggingFaceSaver 增量写入

`StreamingHuggingFaceSaver` 通过 `add_tensors()` 方法实现增量写入：

1. 累积张量到内存缓冲区
2. 当缓冲区大小超过阈值时，自动刷新到磁盘（写入新的 safetensors 分片）
3. `finalize()` 时完成所有收尾工作：
   - 刷新剩余缓冲张量
   - 生成 `model.safetensors.index.json`
   - 复制源模型的 `config.json`、tokenizer 文件、特殊 tokens 文件等
   - NPU 模式下生成 `quant_model_description.json`

---

## 10 完整请求端到端流程

以下是从命令行到最终输出的完整步骤描述：

```
$ python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

**阶段 1：启动**

1. `run.py` 解析命令行参数，调用 `bootstrap_from_path(config_path)`
2. 读取 YAML 文件，校验顶层结构
3. 配置日志系统，保存配置快照到日志目录
4. `parse_config()` 将 YAML 解析为 `EngineConfig`
5. `apply_saver_path_policy()` 推导输出路径
6. `validate_config()` 校验资源引用和必填字段
7. 打印配置信息和 NPUSlim 横幅

**阶段 2：引擎初始化**

8. `SlimEngine.__init__()` 创建 `ResourceManager`，索引资源声明
9. `_build_pipeline()` 遍历 recipe，通过 `TaskRegistry.create()` 创建 `CompressorTask` 实例
10. 任务对象接收 `resource_manager`、model ref、algorithm config、saver config 等

**阶段 3：任务执行**

11. `engine.run()` 调用 `task.execute()`
12. `BaseTask.on_start()`：
    - `rm.acquire_model("@qwen3")` → `ModelRegistry.create("Qwen3", path=..., device_map=...)`
    - `Qwen3SlimModel.__init__()` → `prepare_metadata()` → 加载 tokenizer + config
    - `rm.acquire_dataset("@calib_data")` → 创建数据集 → 封装为 DataLoader
    - `AlgorithmRegistry.create("INT8Dynamic", wbits=8, ...)` → 创建算法实例
    - `SaverRegistry.create("StreamingHuggingFaceSaver", ...)` → 创建 Saver

13. `CompressorTask.run()`：
    - 创建 `ChunkLoader`，根据模型属性配置 `block_name`、`pre/post_module_names`
    - `loader.refresh_index()` → 解析 safetensors 索引，构建层映射
    - 配置 Saver 的源路径和 HF 资产
    - 解析 skip-layer 模式（合并模型内置 + 用户配置的 ignore_layers）
    - 注入算法运行时上下文（model_obj、config、skip_layer_names）
    - `algo.on_start()` → 算法初始化

14. **分块处理循环**（以 streaming 模式为例）：
    - `loader.load_chunk(0)` → 加载 embedding + 前 N 层权重 → 构建 `ChunkContext`
    - 注入 `calib_data` 和 `skip_layer_names` 到 chunk
    - `algo.process_chunk(chunk)` → 对每个层执行 INT8 量化
    - `saver.add_tensors(chunk.all_tensors())` → 增量写入 safetensors 分片
    - `loader.unload_chunk(0)` → 释放内存
    - 重复直到所有 chunk 处理完毕
    - Backfill：补写所有未被 chunk 覆盖的原始张量

15. **收尾**：
    - `algo.on_finish()` → 标记模型已量化
    - `saver.finalize()` → 生成 index.json、复制 config/tokenizer
    - `loader.close()` → 释放所有资源
    - `rm.publish_model_state()` → 更新模型状态

**阶段 4：输出**

16. 输出目录包含：
    - `model-00001-of-0000X.safetensors`（量化后的权重分片）
    - `model.safetensors.index.json`（分片索引）
    - `config.json`（模型配置，已更新量化信息）
    - tokenizer 相关文件（`tokenizer.json`、`tokenizer_config.json` 等）
    - NPU 模式下额外包含 `quant_model_description.json`

17. `SlimEngine.run()` 收集任务结果并返回

---

## 11 BackendHandler 设备管理

`BackendHandler`（`src/npuslim/core/backend.py`）以全局单例 `bh` 的形式提供统一的设备管理：

```python
bh = BackendHandler()  # 自动检测 NPU > CUDA > CPU
```

- **不可变能力检测**：`bh.has_npu`、`bh.has_cuda`、`bh.detected_name`
- **可变放置设备**：`bh.use("npu")` 切换活跃设备，影响 `bh.name`、`bh.device`、`bh.module`
- **运行时操作**：`bh.sync()`、`bh.empty_cache()`、`bh.full_vacuum()`
- **设备映射解析**：`bh.resolve_device_map(device_map)` 将配置中的 `device_map` 值解析为具体的 tensor device 字符串

CompressorTask 在创建 ChunkLoader 时通过 `bh.resolve_device_map()` 确定 tensor 加载设备；NPU 模式下 Saver 要求算法为每个张量提供 `tensor_types` 标注。
