"""Shared registry utilities for NPUSlim plugins.

This module provides generic registration and discovery utilities
that can be used by any plugin (vllm_ascend, transformers, etc.).

Usage:
    from npuslim.plugins.registry import register_patch, discover_and_apply

    @register_patch("some.module.path")
    def patch_something(module):
        module.foo = new_foo
"""

from __future__ import annotations

from typing import Callable

from npuslim.plugins.logging import patch_logger

# Global registry: target_module -> list of patch functions
_PATCH_REGISTRY: dict[str, list[Callable]] = {}
_DISCOVERED_MODULES: set[str] = set()
_APPLIED_PATCHES: set[str] = set()


def register_patch(target: str):
    """Decorator to register a patch for a target module.

    Args:
        target: Full module path to patch
                (e.g., "vllm_ascend.quantization.method_adapters")

    The decorated function receives the imported target module and can
    modify it in place.

    Example:
        @register_patch("vllm_ascend.quantization.method_adapters")
        def patch_process_weight(module):
            original = module.AscendLinearMethod.process_weight
            module.AscendLinearMethod.process_weight = patched_version
    """
    def decorator(func: Callable) -> Callable:
        if target not in _PATCH_REGISTRY:
            _PATCH_REGISTRY[target] = []
        _PATCH_REGISTRY[target].append(func)
        return func
    return decorator


def discover_modules(base_package: str, base_dir: str):
    """Discover and import all Python modules under a base directory.

    This triggers @register_patch and @register_scheme decorators.

    Args:
        base_package: Base package name (e.g., "npuslim.plugins.vllm_ascend")
        base_dir: Base directory path as string
    """
    cache_key = f"{base_package}:{base_dir}"
    if cache_key in _DISCOVERED_MODULES:
        return

    import importlib
    from pathlib import Path

    base_path = Path(base_dir)
    for py_file in base_path.rglob("*.py"):
        if py_file.stem == "__init__":
            continue
        # Convert path to module name
        rel_path = py_file.relative_to(base_path)
        parts = rel_path.with_suffix("").parts
        module_name = f"{base_package}." + ".".join(parts)

        try:
            importlib.import_module(module_name)
            patch_logger.debug(f"Discovered module: {module_name}")
        except ImportError as e:
            patch_logger.warning(f"Failed to import module {module_name}: {e}")

    _DISCOVERED_MODULES.add(cache_key)


def apply_all_patches():
    """Apply all registered patches to their target modules.

    This function is idempotent - patches are only applied once.
    """
    applied = 0
    for target, patches in _PATCH_REGISTRY.items():
        patch_key = target
        if patch_key in _APPLIED_PATCHES:
            continue

        try:
            import importlib
            module = importlib.import_module(target)

            for patch_func in patches:
                try:
                    patch_func(module)
                    patch_logger.debug(
                        f"Applied patch: {patch_func.__name__} -> {target}"
                    )
                    applied += 1
                except Exception as e:
                    patch_logger.warning(
                        f"Failed to apply patch {patch_func.__name__} "
                        f"to {target}: {e}"
                    )
            _APPLIED_PATCHES.add(patch_key)
        except ImportError:
            patch_logger.debug(f"Target module not found, skipping: {target}")

    if applied > 0:
        patch_logger.info(f"Applied {applied} patch(es)")
    return applied


def get_patch_registry():
    """Get the global patch registry (for debugging/testing)."""
    return _PATCH_REGISTRY.copy()
