# 流式保存器（Streaming Saver）内部机制

## 概述

NPUSlim v2 的保存器模块负责将量化后的模型权重以 HuggingFace 兼容格式增量写入磁盘。与传统的"全量加载 -> 全量保存"模式不同，流式保存器采用分片写入策略，支持边量化边落盘，显著降低峰值内存占用。

| 文件 | 说明 |
|------|------|
| `src/npuslim/savers/base_saver.py` | 抽象基类 `BaseSaver`，定义统一接口 |
| `src/npuslim/savers/hf_saver.py` | `StreamingHuggingFaceSaver`，HuggingFace 格式的流式实现 |

## BaseSaver 接口

`BaseSaver` 是所有保存器的抽象基类，定义了三个必须实现的抽象方法：

```python
class BaseSaver(ABC):
    @abstractmethod
    def add_tensor(self, name: str, tensor: torch.Tensor,
                   tensor_type: Optional[str] = None) -> None:
        """添加单个张量到缓冲区。"""

    @abstractmethod
    def add_tensors(self, tensors: Dict[str, torch.Tensor],
                    tensor_types: Optional[Dict[str, str]] = None) -> None:
        """批量添加张量。"""

    @abstractmethod
    def flush(self) -> Optional[str]:
        """将当前缓冲区刷新到磁盘，返回写入的分片文件名。"""

    @abstractmethod
    def finalize(self) -> None:
        """完成保存：写入索引、元数据和辅助文件。"""
```

### 输出目录解析

`resolve_output_dir` 静态方法按优先级解析输出目录：`save_path > output_dir > save_dir`。兼容多种命名习惯的配置文件。

## StreamingHuggingFaceSaver

### 构造参数

```python
class StreamingHuggingFaceSaver(BaseSaver):
    def __init__(
        self,
        save_path=None, output_dir=None, save_dir=None,
        size_threshold=4 * 1024**3,     # 4 GiB 默认分片阈值
        shard_name_pattern="model-{:05d}.safetensors",
        copy_aux_files=True,
        strip_quantization_config_on_npu=True,
        require_tensor_types_on_npu=True,
    ):
```

`size_threshold` 参数支持整数（字节数）和字符串（如 `"4GB"`、`"512MB"`），通过 `_parse_size_to_bytes()` 解析。

### 核心数据结构

```python
self.buffer: Dict[str, torch.Tensor] = {}     # 当前缓冲区
self.buffer_size: int = 0                       # 缓冲区字节大小
self.shard_counter: int = 0                     # 分片计数器
self.weight_map: Dict[str, str] = {}            # 张量名 -> 分片文件名映射
self.tensor_type_map: Dict[str, str] = {}       # 张量名 -> 类型标记映射
```

### 分片写入流程

#### add_tensor —— 添加张量并自动触发 flush

```python
def add_tensor(self, name, tensor, tensor_type=None):
    # NPU 模式强制要求提供 tensor_type
    if bh.has_npu and self.require_tensor_types_on_npu and not tensor_type:
        raise ValueError(...)

    # 如果张量已存在，扣减旧大小
    if name in self.buffer:
        self.buffer_size -= old_tensor.numel() * old_tensor.element_size()

    tensor_size = tensor.numel() * tensor.element_size()

    # 添加前预判：如果会超过阈值，先 flush
    if self.buffer_size + tensor_size > self.size_threshold and self.buffer:
        self.flush()

    # 强制放到 CPU 并确保内存连续
    self.buffer[name] = tensor.cpu().contiguous()
    self.buffer_size += tensor_size

    # 添加后再判：达到阈值立即 flush
    if self.buffer_size >= self.size_threshold:
        self.flush()
```

关键细节：
- **双重阈值检查**：添加前预判和添加后确认
- **张量始终放 CPU**：确保跨设备兼容性
- **NPU 类型强制**：必须提供 `tensor_type`

#### flush —— 将缓冲区写入 safetensors 分片

```python
def flush(self) -> Optional[str]:
    if not self.buffer:
        return None

    # 磁盘空间检查（至少 110% 缓冲区大小）
    total, used, free = shutil.disk_usage(self.output_dir)
    if free < self.buffer_size * 1.1:
        raise IOError(...)

    shard_name = self.shard_name_pattern.format(self.shard_counter)
    save_file(self.buffer, self.output_dir / shard_name)

    # 更新 weight_map
    for name in self.buffer.keys():
        self.weight_map[name] = shard_name

    self.buffer.clear()
    self.buffer_size = 0
    self.shard_counter += 1
    return shard_name
```

### index.json 生成

`finalize()` 调用 `_build_index()` 生成 `model.safetensors.index.json`：

```python
def _build_index(self) -> Dict[str, Any]:
    indexed_shards = {shard for shard in self.weight_map.values()}
    total_size = sum(
        (self.output_dir / shard).stat().st_size
        for shard in indexed_shards
        if (self.output_dir / shard).exists()
    )
    return {
        "metadata": {"total_size": int(total_size)},
        "weight_map": dict(sorted(self.weight_map.items())),
    }
```

### quant_model_description.json（Ascend 专用）

在 NPU 量化场景下，`finalize()` 会检查模型配置中的 `ascend_quant_config`，自动生成 `quant_model_description.json`。此外还会从 `config.json` 中**移除** `quantization_config` 字段，以避免 vLLM-Ascend 运行时的配置冲突。

### 辅助文件拷贝

`_copy_aux_files_from_source()` 从源模型目录拷贝非权重文件到输出目录：

跳过规则：
- 隐藏路径（以 `.` 开头）
- 权重文件后缀（`.safetensors`、`.bin`、`.pt` 等）
- `model.safetensors.index.json`、`optimizer.pt` 等特定文件

如果源路径是 HuggingFace Hub repo ID 而非本地路径，会通过 `huggingface_hub.snapshot_download` 下载非权重文件。

### finalize —— 完整收尾

```python
def finalize(self):
    self.flush()                                    # 1. 刷新剩余缓冲
    index = self._build_index()                     # 2. 写入 index.json
    self._save_hf_assets()                          # 3. 保存 config/tokenizer
    self._copy_aux_files_from_source()              # 4. 拷贝辅助文件
    self._save_ascend_quant_description_if_needed() # 5. 生成 Ascend 量化描述
```

## 完整使用示例

```python
from npuslim.savers.hf_saver import StreamingHuggingFaceSaver

saver = StreamingHuggingFaceSaver(
    output_dir="./outputs/quantized_model",
    size_threshold="2GB",
)

saver.set_source("Qwen/Qwen3-0.6B", model_hub="hf")
saver.set_hf_assets(model_config=config, tokenizer=tokenizer)

# 流式添加张量（通常由 CompressorTask 驱动）
for chunk in chunks:
    for name, tensor in chunk.tensors.items():
        saver.add_tensor(name, tensor, tensor_type="W8A8")

saver.finalize()
```
