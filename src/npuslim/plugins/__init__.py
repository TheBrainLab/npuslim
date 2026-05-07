"""
NPUSlim Plugin System.

Provides automatic registration of NPUSlim quantization methods
with various deployment backends (vLLM, HuggingFace, etc.).
"""

from __future__ import annotations

import importlib
import importlib.util

_REGISTERED = False


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_backend_name() -> str:
    from npuslim.core.backend import bh

    return bh.name


def _register_plugin(package_name: str) -> None:
    module = importlib.import_module(package_name)
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        register_fn()


def register():
    """
    Register all NPUSlim plugins with their respective frameworks.

    Call this once after installing npuslim, or it happens automatically
    via entry points. This function is idempotent - multiple calls are safe.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    if _module_available("vllm"):
        _register_plugin("npuslim.plugins.vllm")

    _register_plugin("npuslim.plugins.transformers")

    if _load_backend_name() == "npu" and _module_available("vllm_ascend"):
        _register_plugin("npuslim.plugins.vllm_ascend")

    if _module_available("speculators"):
        _register_plugin("npuslim.plugins.speculators")

    _REGISTERED = True


__all__ = ["register"]
