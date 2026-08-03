# src/npuslim/__init__.py
"""NPUSlim - Streaming-first quantization framework."""

import importlib.metadata
from pathlib import Path

try:
    __version__ = importlib.metadata.version("npuslim")
except importlib.metadata.PackageNotFoundError:
    try:
        toml_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        if toml_path.exists():
            import re
            content = toml_path.read_text(encoding="utf-8")
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            __version__ = match.group(1) if match else "0.0.0-unknown"
        else:
            __version__ = "0.0.0-dev"
    except Exception:
        __version__ = "0.0.0-dev"


def __getattr__(name: str):
    """Lazily expose heavy runtime objects.

    Importing lightweight subpackages such as ``npuslim.opd`` should not require
    torch, vLLM, or CANN runtime dependencies.  The core engine is loaded only
    when callers request it directly.
    """

    if name == "SlimEngine":
        from npuslim.core import SlimEngine

        return SlimEngine
    raise AttributeError(f"module 'npuslim' has no attribute {name!r}")


__all__ = ["SlimEngine", "__version__"]
