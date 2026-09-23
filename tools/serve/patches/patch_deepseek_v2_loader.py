#!/usr/bin/env python3
"""Patch vLLM deepseek_v2.py expert loader: skip non-local/substring-mismatched chunks.

The expert_params_mapping loop computes `name_mapped = chunk_name.replace(weight_name, param_name)`.
With the npuslim W4A16 packed-mapping patch the "_packed" entry's weight_name
("...proj.weight") is a SUBSTRING of the ".weight_scale"/".weight_offset" chunk
names, and for experts that are NOT local to the current rank the previous
mapping entry's weight_loader returns False. The loop then reaches the packed
entry and `params_dict[name_mapped]` raises KeyError.

Fix: before the lookup, `continue` when the mapped name does not exist in
params_dict (the chunk is then either handled by a later mapping entry or
skipped as non-local). Idempotent: re-running is a no-op.

Target: /vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py
"""
import sys

path = "/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py"
with open(path) as f:
    lines = f.read().split("\n")

target = "param = params_dict[name_mapped]"
needle = "if name_mapped not in params_dict:"

patched = 0
out = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == target:
        ind = line[: len(line) - len(line.lstrip())]
        # Idempotency: check whether the guard `if name_mapped not in
        # params_dict:` already exists at this indent directly above, with
        # only comments/blank/more-indented body lines in between.
        already = False
        j = i - 1
        while j >= 0:
            prev = lines[j]
            if (
                prev.strip() == ""
                or prev.strip().startswith("#")
                or prev.startswith(ind + "    ")
            ):
                j -= 1
                continue
            if prev.startswith(ind) and prev.strip() == needle:
                already = True
            break
        if not already:
            out.append(f"{ind}if name_mapped not in params_dict:")
            out.append(f"{ind}    # Substring-based expert mapping can over-match a")
            out.append(f"{ind}    # scale/offset chunk (e.g. the \"_packed\" entry")
            out.append(f"{ind}    # matches \"...proj.weight_offset\"), or the expert")
            out.append(f"{ind}    # is not local to this rank. Skip this mapping entry.")
            out.append(f"{ind}    continue")
            patched += 1
        out.append(line)
    else:
        out.append(line)
    i += 1

if patched == 0:
    print("NO-OP: guard already present or target not found")
    sys.exit(0)

with open(path, "w") as f:
    f.write("\n".join(out))
print(f"PATCHED: inserted {patched} guard(s)")
