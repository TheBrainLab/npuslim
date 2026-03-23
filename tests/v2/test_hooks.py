# tests/v2/test_hooks.py
import pytest
from npuslim.v2.hooks import (
    HookType,
    HookInfo,
    HookRegistry,
    HookDispatcher,
    register_hook,
)


def test_hook_type_enum():
    """Test HookType enum has all expected values."""
    assert HookType.ON_START.value == "on_start"
    assert HookType.ON_CHUNK_ENTER.value == "on_chunk_enter"
    assert HookType.ON_TENSOR_EMIT.value == "on_tensor_emit"


def test_register_hook_decorator():
    """Test @register_hook decorator registers hooks."""
    # Clear registry first
    HookRegistry.clear()

    @register_hook(HookType.ON_CHUNK_EXIT, priority=10)
    def my_test_hook(context, **kwargs):
        return {"executed": True}

    # Verify registration
    hooks = HookRegistry.get_hooks_by_type(HookType.ON_CHUNK_EXIT)
    assert len(hooks) == 1
    assert hooks[0].name == "my_test_hook"
    assert hooks[0].priority == 10


def test_hook_dispatcher():
    """Test HookDispatcher executes hooks in priority order."""
    HookRegistry.clear()

    execution_order = []

    @register_hook(HookType.ON_START, priority=1)
    def low_priority_hook(context, **kwargs):
        execution_order.append("low")

    @register_hook(HookType.ON_START, priority=10)
    def high_priority_hook(context, **kwargs):
        execution_order.append("high")

    dispatcher = HookDispatcher(HookRegistry.get_hooks_by_type(HookType.ON_START))
    dispatcher.dispatch(None)

    # Higher priority should execute first
    assert execution_order == ["high", "low"]
