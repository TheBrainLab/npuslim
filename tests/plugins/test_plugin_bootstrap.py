from __future__ import annotations

import importlib
import sys
import types


def test_plugin_register_skips_unavailable_optional_backends(monkeypatch):
    import npuslim.plugins as plugins

    plugins = importlib.reload(plugins)

    calls: list[str] = []

    monkeypatch.setattr(plugins, "_module_available", lambda name: name == "vllm")
    monkeypatch.setattr(
        plugins,
        "_load_backend_name",
        lambda: "cpu",
    )

    vllm_module = types.ModuleType("npuslim.plugins.vllm")
    vllm_module.register = lambda: calls.append("vllm")
    transformers_module = types.ModuleType("npuslim.plugins.transformers")
    transformers_module.register = lambda: calls.append("transformers")

    monkeypatch.setitem(sys.modules, "npuslim.plugins.vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "npuslim.plugins.transformers", transformers_module)

    plugins.register()

    assert calls == ["vllm", "transformers"]


def test_plugin_register_loads_npu_and_speculators_when_available(monkeypatch):
    import npuslim.plugins as plugins

    plugins = importlib.reload(plugins)

    calls: list[str] = []

    monkeypatch.setattr(
        plugins,
        "_module_available",
        lambda name: name in {"vllm", "vllm_ascend", "speculators"},
    )
    monkeypatch.setattr(
        plugins,
        "_load_backend_name",
        lambda: "npu",
    )

    modules = {
        "npuslim.plugins.vllm": types.ModuleType("npuslim.plugins.vllm"),
        "npuslim.plugins.vllm_ascend": types.ModuleType("npuslim.plugins.vllm_ascend"),
        "npuslim.plugins.transformers": types.ModuleType("npuslim.plugins.transformers"),
        "npuslim.plugins.speculators": types.ModuleType("npuslim.plugins.speculators"),
    }
    modules["npuslim.plugins.vllm"].register = lambda: calls.append("vllm")
    modules["npuslim.plugins.vllm_ascend"].register = lambda: calls.append("vllm_ascend")
    modules["npuslim.plugins.transformers"].register = lambda: calls.append("transformers")
    modules["npuslim.plugins.speculators"].register = lambda: calls.append("speculators")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    plugins.register()

    assert calls == ["vllm", "transformers", "vllm_ascend", "speculators"]


def test_plugin_registry_import_does_not_require_vllm(monkeypatch):
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "vllm":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    sys.modules.pop("npuslim.plugins.registry", None)

    module = importlib.import_module("npuslim.plugins.registry")

    assert module.logger is not None
