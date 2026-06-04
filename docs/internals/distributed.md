# 分布式（Distributed）模块内部机制

## 概述

`distributed` 模块位于 `src/npuslim/distributed/`，为 NPUSlim v2 提供统一的分布式执行支持。它封装了三种主流分布式后端（HuggingFace Accelerate、原生 torch.distributed、DeepSpeed），对外暴露一致的 API，使上层引擎和任务无需关心底层差异。

## 配置定义

分布式配置定义在 `src/npuslim/config/schema.py` 中：

```python
class DistributedBackend(Enum):
    NONE = "none"                      # 单进程模式
    ACCELERATE = "accelerate"          # HuggingFace Accelerate
    TORCH_DISTRIBUTED = "torch_distributed"  # 原生 torch.distributed
    DEEPSPEED = "deepspeed"            # DeepSpeed
```

默认值为单进程模式（`NONE`），即不启用任何分布式功能。

## DistributedManager 核心类

### 初始化流程

```python
class DistributedManager:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self._accelerator = None
        self._model = None
        self._optimizer = None
        self._setup()
```

### 三种后端的初始化

#### Accelerate 后端

```python
def _setup_accelerate(self):
    from accelerate import Accelerator
    self._accelerator = Accelerator(
        mixed_precision=self.config.mixed_precision,
        gradient_accumulation_steps=self.config.gradient_accumulation_steps,
    )
```

#### torch.distributed 后端

```python
def _setup_torch_distributed(self):
    import torch.distributed as dist

    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", self.config.rank))
        world_size = int(os.environ.get("WORLD_SIZE", self.config.world_size))
        local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))

        dist.init_process_group(
            backend=self.config.backend_init_method,  # 默认 "nccl"
            rank=rank, world_size=world_size,
        )
        bh.set_device(local_rank)
```

关键细节：
- 环境变量优先于配置值（兼容 `torchrun` 启动方式）
- 调用 `bh.set_device(local_rank)` 绑定当前进程的物理设备

#### DeepSpeed 后端

DeepSpeed 采用惰性初始化策略，因为 `deepspeed.initialize()` 需要同时接收模型和配置。

### 属性接口

```python
@property
def is_distributed(self) -> bool:
    """是否运行在分布式模式。"""

@property
def is_main_process(self) -> bool:
    """当前进程是否为主进程（rank 0）。"""

@property
def world_size(self) -> int:
    """总进程数。"""

@property
def rank(self) -> int:
    """当前进程的全局排名。"""

@property
def local_rank(self) -> int:
    """当前进程在节点内的排名。"""
```

### 模型/优化器封装

```python
def prepare_model(self, model, optimizer=None, dataloader=None,
                  lr_scheduler=None) -> tuple:
    """为分布式执行封装模型和优化器。"""
```

DDP 封装通过 `BackendHandler` 确定设备索引格式，支持 CUDA 和 NPU 的 `DistributedDataParallel`。

### 集合通信

提供四种集合通信原语：

| 方法 | 说明 |
|------|------|
| `barrier()` | 同步所有进程 |
| `gather(tensor)` | 从所有进程收集张量 |
| `reduce(tensor, op)` | 跨进程规约张量（sum/mean/max/min） |
| `broadcast(tensor, source)` | 从 source 进程广播张量 |

### 上下文管理器

```python
@contextmanager
def main_process_first(self):
    """主进程先执行，其余进程等待后执行。"""
    if self.is_main_process:
        yield
    self.barrier()
    if not self.is_main_process:
        yield
    self.barrier()
```

典型场景：主进程下载模型/数据集，其余进程等待复用。

### 资源清理

```python
def destroy(self) -> None:
    """清理分布式资源。"""
    if self.config.backend == DistributedBackend.TORCH_DISTRIBUTED:
        dist.destroy_process_group()
```

## 与引擎的协作方式

`DistributedManager` 通过配置驱动与 `SlimEngine` 协作。引擎根据配置创建 `DistributedManager` 实例，并将其传递给需要分布式能力的任务。

在量化管线中，分布式的主要用途包括：
1. **多卡并行量化**：不同 rank 处理不同的模型分片
2. **校准数据分布式加载**：每个 rank 加载不同的校准样本
3. **模型保存同步**：仅主进程执行 `finalize` 写入操作

## 代码示例

```python
from npuslim.config.schema import DistributedConfig, DistributedBackend
from npuslim.distributed import DistributedManager

config = DistributedConfig(
    backend=DistributedBackend.TORCH_DISTRIBUTED,
    world_size=4,
    backend_init_method="nccl",
)
manager = DistributedManager(config)

print(f"Rank {manager.rank}/{manager.world_size}")

# 封装模型
model, optimizer, dl, scheduler = manager.prepare_model(model, optimizer, dataloader)

# 主进程保存结果
if manager.is_main_process:
    saver.finalize()

manager.barrier()
manager.destroy()
```
