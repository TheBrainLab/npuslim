# BackendHandler 内部机制

## 概述

`BackendHandler` 是 NPUSlim v2 的统一后端抽象层，位于 `src/npuslim/core/backend.py`。它为运行时、流式写入和分布式模块提供统一的设备管理接口，屏蔽了 CPU / CUDA / NPU 三种硬件平台之间的差异。

BackendHandler 的设计遵循一个核心原则：**不可变能力（Immutable Capability）与可变部署（Mutable Placement）的分离**。硬件能力在实例化时即被锁定，而活跃设备的切换则可在运行时动态进行。

## 设计哲学：不可变能力 vs 可变部署

### 不可变能力（Capability）

不可变属性反映的是**物理硬件的实际存在状态**，在对象生命周期内不会发生变化。这些属性用于功能分支判断、输出格式选择和插件注册等场景。

```python
class BackendHandler:
    def __init__(self) -> None:
        # 不可变：硬件检测，仅在构造时执行一次
        self._has_npu: bool = hasattr(torch, "npu") and torch.npu.is_available()
        self._has_cuda: bool = torch.cuda.is_available()
        self._detected_name: str = self._auto_detect()
```

三个不可变属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `detected_name` | `str` | 自动检测到的最佳设备名称（`"npu"` / `"cuda"` / `"cpu"`） |
| `has_npu` | `bool` | 系统是否存在 NPU 硬件 |
| `has_cuda` | `bool` | 系统是否存在 CUDA 硬件 |

### 可变部署（Placement）

可变属性控制的是**张量实际存放的设备位置**，可通过 `use()` 方法在运行时动态切换。

```python
class BackendHandler:
    def __init__(self) -> None:
        # 可变：活跃设备状态（初始化为自动检测值）
        self._name: str = self._detected_name
        self._device: torch.device = torch.device(self._name)
        self._module = self._resolve_module(self._name)
```

这种分离意味着即使在 CUDA 环境下将部署设备切换到 CPU（例如为了离线调试），`has_cuda` 仍然返回 `True`，NPU 插件注册逻辑不会被错误地跳过。

## 硬件检测逻辑

自动检测在构造时执行一次，优先级为 **NPU > CUDA > CPU**：

```python
def _auto_detect(self) -> str:
    if self._has_npu:
        return "npu"
    if self._has_cuda:
        return "cuda"
    return "cpu"
```

NPU 的检测条件有两层：
1. `hasattr(torch, "npu")` —— 确保 PyTorch 编译了 NPU 支持
2. `torch.npu.is_available()` —— 确保驱动和 CANN 运行时可用

## 设备切换：use() 方法

```python
def use(self, device_name: str) -> None:
    normalized = device_name.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"

    if normalized not in ("cpu", "cuda", "npu"):
        raise ValueError(...)
    if normalized == "npu" and not self._has_npu:
        raise RuntimeError("NPU requested but not available")
    if normalized == "cuda" and not self._has_cuda:
        raise RuntimeError("CUDA requested but not available")

    self._name = normalized
    self._device = torch.device(normalized)
    self._module = self._resolve_module(normalized)
```

关键行为：
- `"gpu"` 会被规范化为 `"cuda"`
- 切换前会校验目标设备的可用性
- **不会修改**不可变属性

## 设备映射解析

`resolve_device_map()` 将配置文件中的 `device_map` 字段解析为具体的运行时设备字符串。支持字符串、整数和字典三种输入形式：

```python
def resolve_device_map(self, device_map: Any, default: str = "cpu") -> str:
    if isinstance(device_map, str):
        # "auto" / "balanced" -> 当前活跃设备
        # "cuda" / "gpu" -> "cuda:0"
        # "npu" -> "npu:0"
        # "cpu" / "disk" -> "cpu"
    if isinstance(device_map, int):
        # 视为 CUDA 设备索引 -> "cuda:{device_map}"
    if isinstance(device_map, dict):
        # 遍历值，找到第一个非 CPU 映射
```

## 张量迁移与设备操作

### sync —— 设备同步

```python
def sync(self, device: str | None = None) -> None:
    target = (device or self.default_device_str()).lower()
    if target.startswith("npu"):
        torch.npu.synchronize()
    if target.startswith("cuda"):
        torch.cuda.synchronize()
```

### empty_cache —— 清空缓存

```python
def empty_cache(self, device: str | None = None) -> None:
    target = (device or self.default_device_str()).lower()
    if target.startswith("npu"):
        torch.npu.empty_cache()
    if target.startswith("cuda"):
        torch.cuda.empty_cache()
```

### full_vacuum —— 完整清理

```python
def full_vacuum(self, device: str | None = None) -> None:
    self.sync(device=device)
    gc.collect()
    self.empty_cache(device=device)
```

`full_vacuum` 执行同步 -> Python GC -> 设备缓存释放的完整链路，通常在处理完一个 chunk 后调用以回收内存。

### NPU 特殊处理

1. **设备索引**：NPU 使用 `npu:{local_rank}` 格式，与 CUDA 对称
2. **模块解析**：通过 `hasattr(torch, "npu")` 防御性检查
3. **set_device**：调用 `torch.npu.set_device(local_rank)` 设置当前进程使用的 NPU 设备

## 全局单例

模块级创建一个全局单例 `bh`，供整个框架共享：

```python
bh = BackendHandler()
```

所有需要设备信息的模块均通过 `from npuslim.core.backend import bh` 引用该单例，确保全局设备状态一致。

## 代码示例

```python
from npuslim.core.backend import bh

# 查询硬件能力（不可变）
print(f"NPU 可用: {bh.has_npu}")
print(f"CUDA 可用: {bh.has_cuda}")

# 运行时切换设备
bh.use("npu")

# 在算法中使用
tensor = chunk.tensor.to(bh.device)
bh.full_vacuum()
```
