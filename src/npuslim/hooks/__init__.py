"""Hook system for NPUSlim."""
from npuslim.hooks.hooks import (
    HookType,
    HookInfo,
    HookRegistry,
    HookDispatcher,
    register_hook,
)

__all__ = ["HookType", "HookInfo", "HookRegistry", "HookDispatcher", "register_hook"]
