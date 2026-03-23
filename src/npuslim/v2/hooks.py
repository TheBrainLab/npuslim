"""Hook system for NPUSlim v2."""
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
from enum import Enum


class HookType(Enum):
    """Lifecycle hook types."""
    # Pipeline lifecycle
    ON_START = "on_start"
    ON_FINISH = "on_finish"
    # Task lifecycle
    ON_TASK_START = "on_task_start"
    ON_TASK_FINISH = "on_task_finish"
    # Algorithm lifecycle
    ON_ALGORITHM_START = "on_algorithm_start"
    ON_ALGORITHM_FINISH = "on_algorithm_finish"
    # Chunk lifecycle
    ON_CHUNK_ENTER = "on_chunk_enter"
    ON_CHUNK_EXIT = "on_chunk_exit"
    # Layer lifecycle
    ON_LAYER_ENTER = "on_layer_enter"
    ON_LAYER_EXIT = "on_layer_exit"
    # Step lifecycle
    ON_STEP_ENTER = "on_step_enter"
    ON_STEP_EXIT = "on_step_exit"
    # Streaming lifecycle
    ON_TENSOR_EMIT = "on_tensor_emit"
    ON_TENSOR_FLUSH = "on_tensor_flush"


@dataclass
class HookInfo:
    """Information about a registered hook."""
    name: str
    func: Callable
    hook_type: HookType
    priority: int = 0  # Higher priority hooks run first
    description: str = ""


class HookRegistry:
    """Global registry for hooks."""
    _hooks: Dict[str, HookInfo] = {}

    @classmethod
    def register(cls, hook_type: HookType, name: Optional[str] = None, priority: int = 0):
        """Decorator to register a hook."""
        def decorator(func: Callable) -> Callable:
            hook_name = name or func.__name__
            if hook_name in cls._hooks:
                raise ValueError(f"Hook '{hook_name}' already registered")
            cls._hooks[hook_name] = HookInfo(
                name=hook_name,
                func=func,
                hook_type=hook_type,
                priority=priority,
            )
            return func
        return decorator

    @classmethod
    def get_hooks_by_type(cls, hook_type: HookType) -> List[HookInfo]:
        """Get all hooks of a specific type."""
        return [h for h in cls._hooks.values() if h.hook_type == hook_type]

    @classmethod
    def clear(cls):
        """Clear all registered hooks (for testing)."""
        cls._hooks.clear()


# Convenience alias
register_hook = HookRegistry.register


class HookDispatcher:
    """Dispatches hooks in priority order."""

    def __init__(self, hooks: List[HookInfo]):
        self.hooks = hooks

    def dispatch(self, context: Any, **kwargs) -> List[Any]:
        """Execute all hooks, return results."""
        # Sort by priority (higher priority first)
        sorted_hooks = sorted(self.hooks, key=lambda h: h.priority, reverse=True)

        results = []
        for hook_info in sorted_hooks:
            try:
                result = hook_info.func(context, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but continue
                import traceback
                traceback.print_exc()
        return results
