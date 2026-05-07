from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_optional_dependency_groups_for_runtime_backends():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional = project["optional-dependencies"]

    assert "lm-eval[api]" not in dependencies
    assert "evalscope[perf]" not in dependencies
    assert "vllm" not in dependencies
    assert "vllm-ascend" not in dependencies
    assert "speculators" not in dependencies

    assert optional["eval"] == ["lm-eval[api]", "evalscope[perf]"]
    assert optional["vllm"] == ["vllm"]
    assert optional["npu"] == ["vllm-ascend"]
    assert optional["speculators"] == ["speculators"]
    assert set(optional["all"]) >= {
        "lm-eval[api]",
        "evalscope[perf]",
        "vllm",
        "vllm-ascend",
        "speculators",
    }
