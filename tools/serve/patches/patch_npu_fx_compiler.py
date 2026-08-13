#!/usr/bin/env python3
"""Patch torch_npu compilers: guard enable_view_optimize assignment.

When the compiled graph contains `npu.npu_weight_quant_batchmatmul` (the W4A16
quantized matmul), torch_npu tries to set
`experimental_config.enable_view_optimize = False`. That option only exists in
torchair's _ExperimentalConfig, not in npugraph_ex's, so the assignment raises
ValueError and aborts vLLM engine init. Wrap it in try/except.

Idempotent: re-running is a no-op. Applies to both npugraph_ex and torchair
compiler files.
"""
import sys

FILES = [
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py",
    "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/dynamo/torchair/npu_fx_compiler.py",
]

for path in FILES:
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"SKIP (missing): {path}")
        continue

    needle = "self.config.experimental_config.enable_view_optimize = False"
    if needle not in src:
        print(f"NO-OP (pattern not found): {path}")
        continue

    # Extract the surrounding indentation.
    idx = src.index(needle)
    line_start = src.rfind("\n", 0, idx) + 1
    ind = src[line_start:idx]

    if src.count("try:") > 0 and src.rindex("try:", 0, idx) > src.rfind("\n", 0, line_start):
        # A try: exists somewhere above; assume already guarded.
        print(f"NO-OP (already guarded?): {path}")
        continue

    old = f"{ind}{needle}\n"
    new = (
        f"{ind}try:\n"
        f"{ind}    {needle}\n"
        f"{ind}except (ValueError, AttributeError):\n"
        f"{ind}    pass  # option only exists in torchair's experimental config\n"
    )
    assert src.count(old) == 1, f"unexpected occurrence count {src.count(old)} in {path}"
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print(f"PATCHED: {path}")
