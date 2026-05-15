"""Shared registry utilities for NPUSlim plugins.

This module provides generic registration and discovery utilities
that can be used by any plugin (vllm_ascend, transformers, etc.).

Usage:
    from npuslim.plugins.registry import (
        package_version_range,
        register_patch,
    )

    @register_patch(target="some.module.path")
    def patch_something(module):
        module.foo = new_foo

    @register_patch(
        target="some.other.module",
        condition=package_version_range("somepkg", max_version="1.2.3"),
    )
    def patch_only_old_versions(module):
        ...

    @register_patch(
        registrar=some_framework_register("custom_name"),
        condition=package_version_range("somepkg", max_version="1.2.3"),
    )
    class CustomImpl:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from packaging.version import InvalidVersion, Version

from npuslim.plugins.logging import patch_logger

PatchConditionResult = bool | tuple[bool, str]
PatchCondition = Callable[[Any], PatchConditionResult]
Registrar = Callable[[Any], Any]


@dataclass(frozen=True)
class PatchSpec:
    func: Callable
    condition: PatchCondition | None = None


# Global registry: target_module -> list of patch specs
_PATCH_REGISTRY: dict[str, list[PatchSpec]] = {}
_DISCOVERED_MODULES: set[str] = set()
_APPLIED_PATCHES: set[str] = set()


def _is_valid_module_parts(parts: tuple[str, ...]) -> bool:
    return all(part.isidentifier() for part in parts)


def _object_name(obj: Any) -> str:
    return getattr(obj, "__name__", repr(obj))


def _evaluate_condition(
    condition: PatchCondition | None,
    subject: Any,
) -> tuple[bool, str | None]:
    if condition is None:
        return True, None

    return _normalize_condition_result(condition(subject))


def _skip_reason(reason: str | None) -> str:
    return reason or "condition returned False"


def package_version_range(
    package_name: str,
    *,
    min_version: str | None = None,
    max_version: str | None = None,
    include_min: bool = True,
    include_max: bool = False,
    version_attr: str = "__version__",
) -> PatchCondition:
    """Build a patch condition constrained by an installed package version.

    Args:
        package_name: Top-level package name to inspect.
        min_version: Inclusive or exclusive lower bound.
        max_version: Inclusive or exclusive upper bound.
        include_min: Whether ``min_version`` is inclusive.
        include_max: Whether ``max_version`` is inclusive.
        version_attr: Attribute used to read the package version.
    """
    if min_version is None and max_version is None:
        raise ValueError("At least one of min_version or max_version is required")

    min_v = Version(min_version) if min_version is not None else None
    max_v = Version(max_version) if max_version is not None else None

    def condition(_module: Any) -> PatchConditionResult:
        import importlib
        from importlib.metadata import PackageNotFoundError, version as dist_version

        package = importlib.import_module(package_name)
        raw_version = getattr(package, version_attr, None)
        if raw_version is None:
            try:
                raw_version = dist_version(package_name)
            except PackageNotFoundError:
                return False, (
                    f"{package_name}.{version_attr} is unavailable and package "
                    "metadata version was not found"
                )

        try:
            current = Version(str(raw_version))
        except InvalidVersion:
            return False, (
                f"{package_name}.{version_attr}={raw_version!r} is not a valid version"
            )

        if min_v is not None:
            if include_min:
                min_ok = current >= min_v
                min_text = f">= {min_v}"
            else:
                min_ok = current > min_v
                min_text = f"> {min_v}"
            if not min_ok:
                return False, f"{package_name} {current} does not satisfy {min_text}"

        if max_v is not None:
            if include_max:
                max_ok = current <= max_v
                max_text = f"<= {max_v}"
            else:
                max_ok = current < max_v
                max_text = f"< {max_v}"
            if not max_ok:
                return False, f"{package_name} {current} does not satisfy {max_text}"

        return True

    return condition


def _normalize_condition_result(
    result: PatchConditionResult,
) -> tuple[bool, str | None]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "Patch condition tuple results must be shaped as (bool, reason)"
            )
        return bool(result[0]), str(result[1])

    return bool(result), None


def register_patch(
    *,
    target: str | None = None,
    registrar: Registrar | None = None,
    condition: PatchCondition | None = None,
):
    """Decorator to register a patch for a target module.

    Args:
        target: Full module path to patch
                (e.g., "vllm_ascend.quantization.method_adapters")
        registrar: Optional external registration decorator returned by
                   framework APIs such as ``CustomOp.register_oot(...)``.
                   When provided, registration is conditionally executed
                   immediately at import time.
        condition: Optional callable evaluated during unified patch
                   application. If omitted, the patch always applies.

    The decorated function receives the imported target module and can
    modify it in place.

    Example:
        @register_patch(target="vllm_ascend.quantization.method_adapters")
        def patch_process_weight(module):
            original = module.AscendLinearMethod.process_weight
            module.AscendLinearMethod.process_weight = patched_version
    """
    if target is None and registrar is None:
        raise ValueError("register_patch requires either target or registrar")
    if target is not None and registrar is not None:
        raise ValueError("register_patch target and registrar are mutually exclusive")

    def decorator(obj: Callable) -> Callable:
        if registrar is not None:
            should_apply, reason = _evaluate_condition(condition, obj)
            if not should_apply:
                patch_logger.info(
                    f"Skipped registrar: {_object_name(obj)} "
                    f"({_skip_reason(reason)})"
                )
                return obj

            registered_obj = registrar(obj)
            patch_logger.success(f"Applied registrar: {_object_name(obj)}")
            return registered_obj

        assert target is not None
        if target not in _PATCH_REGISTRY:
            _PATCH_REGISTRY[target] = []
        _PATCH_REGISTRY[target].append(PatchSpec(func=obj, condition=condition))
        return obj

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
        if not _is_valid_module_parts(parts):
            patch_logger.info(f"Skipped non-module file during discovery: {py_file}")
            continue
        module_name = f"{base_package}." + ".".join(parts)

        try:
            importlib.import_module(module_name)
            patch_logger.info(f"Discovered module: {module_name}")
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

            for patch_spec in patches:
                try:
                    should_apply, reason = _evaluate_condition(
                        patch_spec.condition, module
                    )
                    if not should_apply:
                        patch_logger.info(
                            f"Skipped patch: {patch_spec.func.__name__} -> "
                            f"{target} ({_skip_reason(reason)})"
                        )
                        continue

                    patch_spec.func(module)
                    patch_logger.success(
                        f"Applied patch: {patch_spec.func.__name__} -> {target}"
                    )
                    applied += 1
                except Exception as e:
                    patch_logger.warning(
                        f"Failed to apply patch {patch_spec.func.__name__} "
                        f"to {target}: {e}"
                    )
            _APPLIED_PATCHES.add(patch_key)
        except ImportError:
            patch_logger.info(f"Target module not found, skipping: {target}")

    if applied > 0:
        patch_logger.success(f"Applied {applied} patch(es)")
    return applied


def get_patch_registry():
    """Get the global patch registry (for debugging/testing)."""
    return _PATCH_REGISTRY.copy()
