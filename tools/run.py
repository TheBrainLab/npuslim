"""NPUSlim CLI entry point (compat wrapper).

Kept for backward compatibility with existing invocations
(``python tools/run.py -c configs/...yaml``). The canonical CLI
implementation lives in ``npuslim.cli.__main__`` (the ``npuslim``
console-script entry point declared in ``pyproject.toml``).
"""

from __future__ import annotations

from npuslim.cli.__main__ import main

if __name__ == "__main__":
    main()
