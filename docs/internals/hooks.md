# 钩子（Hooks）系统内部机制

## 概述

NPUSlim v2 的钩子系统提供了一个轻量级、可扩展的生命周期事件机制。它允许用户在管线的各个阶段注入自定义逻辑（例如日志记录、指标采集、张量检查等），而无需修改核心框架代码。

| 组件 | 职责 |
|------|------|
| `HookType` | 枚举类型，定义所有可用的钩子点 |
| `HookRegistry` | 全局注册表，存储和管理已注册的钩子 |
| `HookDispatcher` | 调度器，按优先级执行钩子函数 |

## HookType 枚举

`HookType` 枚举定义了框架中所有可用的生命周期钩子点：

```python
class HookType(Enum):
    # 管线生命周期
    ON_START = "on_start"
    ON_FINISH = "on_finish"

    # 任务生命周期
    ON_TASK_START = "on_task_start"
    ON_TASK_FINISH = "on_task_finish"

    # 算法生命周期
    ON_ALGORITHM_START = "on_algorithm_start"
    ON_ALGORITHM_FINISH = "on_algorithm_finish"

    # Chunk 生命周期
    ON_CHUNK_ENTER = "on_chunk_enter"
    ON_CHUNK_EXIT = "on_chunk_exit"

    # 层生命周期
    ON_LAYER_ENTER = "on_layer_enter"
    ON_LAYER_EXIT = "on_layer_exit"

    # 流式写入生命周期
    ON_TENSOR_EMIT = "on_tensor_emit"
    ON_TENSOR_FLUSH = "on_tensor_flush"
```

### 管线阶段与钩子对应关系

```
SlimEngine.run()
├── ON_START
├── Task.execute()
│   ├── ON_TASK_START
│   ├── Algorithm.on_start()
│   │   ├── ON_ALGORITHM_START
│   │   ├── for chunk in chunks:
│   │   │   ├── ON_CHUNK_ENTER
│   │   │   ├── for layer in chunk.layers:
│   │   │   │   ├── ON_LAYER_ENTER
│   │   │   │   └── ON_LAYER_EXIT
│   │   │   ├── ON_TENSOR_EMIT
│   │   │   └── ON_CHUNK_EXIT
│   │   └── ON_ALGORITHM_FINISH
│   ├── ON_TENSOR_FLUSH
│   └── ON_TASK_FINISH
└── ON_FINISH
```

## HookInfo 数据类

```python
@dataclass
class HookInfo:
    name: str            # 钩子唯一标识名
    func: Callable       # 钩子函数
    hook_type: HookType  # 钩子类型
    priority: int = 0    # 优先级，数值越高越先执行
    description: str = "" # 描述信息
```

## HookRegistry 注册表

`HookRegistry` 是一个类级别的全局注册表，管理所有已注册的钩子。

### 注册：register 装饰器

```python
class HookRegistry:
    _hooks: Dict[str, HookInfo] = {}

    @classmethod
    def register(cls, hook_type: HookType, name: Optional[str] = None,
                 priority: int = 0):
        """装饰器：注册一个钩子函数。"""
        def decorator(func: Callable) -> Callable:
            hook_name = name or func.__name__
            if hook_name in cls._hooks:
                raise ValueError(f"Hook '{hook_name}' already registered")
            cls._hooks[hook_name] = HookInfo(
                name=hook_name, func=func,
                hook_type=hook_type, priority=priority,
            )
            return func
        return decorator
```

特点：
- **去重保护**：同名钩子只能注册一次
- **灵活命名**：可通过 `name` 参数显式指定钩子名
- **装饰器模式**：注册后返回原函数

### 查询与清除

```python
@classmethod
def get_hooks_by_type(cls, hook_type: HookType) -> List[HookInfo]:
    """获取指定类型的所有已注册钩子。"""

@classmethod
def clear(cls):
    """清除所有已注册的钩子（用于测试）。"""
```

## HookDispatcher 调度器

```python
class HookDispatcher:
    def dispatch(self, context: Any, **kwargs) -> List[Any]:
        """按优先级执行所有钩子，返回结果列表。"""
        sorted_hooks = sorted(self.hooks, key=lambda h: h.priority, reverse=True)
        results = []
        for hook_info in sorted_hooks:
            try:
                result = hook_info.func(context, **kwargs)
                results.append(result)
            except Exception as e:
                logger.exception(f"Hook '{hook_info.name}' failed")
        return results
```

关键行为：
- **优先级排序**：`priority` 值越大的钩子越先执行
- **异常隔离**：单个钩子的异常不会中断其他钩子
- **结果收集**：返回所有钩子的返回值列表

## 自定义钩子示例

### 量化进度日志

```python
from npuslim.hooks import HookType, register_hook

@register_hook(HookType.ON_CHUNK_EXIT, name="chunk_progress_logger")
def log_chunk_progress(context, **kwargs):
    chunk_id = getattr(context, "chunk_id", "?")
    total = getattr(context, "total_chunks", "?")
    print(f"[Progress] Chunk {chunk_id}/{total} completed")
```

### 张量统计监控

```python
@register_hook(HookType.ON_TENSOR_EMIT, name="tensor_stats_monitor", priority=10)
def monitor_tensor_stats(context, **kwargs):
    tensor = getattr(context, "tensor", None)
    if tensor is None:
        return None
    return {
        "name": getattr(context, "name", "unknown"),
        "shape": tuple(tensor.shape),
        "mean": tensor.float().mean().item(),
        "std": tensor.float().std().item(),
    }
```

### 内存使用追踪

```python
from npuslim.core.backend import bh

@register_hook(HookType.ON_LAYER_EXIT, name="memory_tracker", priority=5)
def track_memory(context, **kwargs):
    if bh.name == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[Memory] Layer done: allocated={allocated:.2f}GiB")
```

## 设计注意事项

1. **全局状态**：`HookRegistry._hooks` 是类变量，全局共享。测试中应使用 `clear()` 清理状态
2. **异常安全**：`HookDispatcher` 捕获单个钩子的异常，不会传播到调用方
3. **性能开销**：钩子在热路径上执行，应避免耗时操作
4. **线程安全**：当前实现适用于单线程的量化管线场景
