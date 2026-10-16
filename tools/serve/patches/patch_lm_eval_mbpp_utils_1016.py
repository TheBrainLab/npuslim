#!/usr/bin/env python3
"""Patch lm_eval's mbpp metric loader so the long-context mbpp eval works offline.

Problem (2026-10-16, cost 3 failed runs on .57/.47):
the container's `lm_eval/tasks/mbpp/utils.py` loads the code_eval metric by BARE
NAME:

    pass_at_k = hf_evaluate.load("code_eval")

With HF_HUB_OFFLINE=1 and no egress from the container this falls back to the
HuggingFace Hub -> ~190 s TCP timeout -> FileNotFoundError at metric-import time
-> the whole lm_eval run dies before sending a single request.  Some containers
(.106) already carry the fixed version; .57/.47/.135 did not.

Fix: install the known-good utils.py (asset) which loads the metric from a local
absolute path and adds EXTRACT_PATCH_1010B fence extraction (needed because GLM
answers come back inside markdown fences).

Idempotent: NO-OP when the target already matches the asset.
"""
import glob
import hashlib
import os
import shutil
import sys

ASSET = "/llm_infer_workspace/ylt/npuslim/tools/serve/patches/assets/lm_eval_mbpp_utils_1016.py"
TARGET_GLOB = "/usr/local/python3*/lib/python3*/site-packages/lm_eval/tasks/mbpp/utils.py"
MARKERS = ("qwen3_deploy", "EXTRACT_PATCH_1010B")


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    try:
        with open(ASSET) as fh:
            asset_src = fh.read()
    except FileNotFoundError:
        print(f"ERROR (asset missing): {ASSET}")
        return 1
    missing = [m for m in MARKERS if m not in asset_src]
    if missing:
        print(f"ERROR (asset lacks markers {missing}): {ASSET}")
        return 1

    targets = sorted(glob.glob(TARGET_GLOB))
    if not targets:
        print(f"SKIP (target not found): {TARGET_GLOB}")
        return 1

    asset_md5 = md5(ASSET)
    for path in targets:
        if md5(path) == asset_md5:
            print(f"NO-OP (already patched): {path}")
            continue
        bak = path + ".bak_1016"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        shutil.copy2(ASSET, path)
        print(f"PATCHED: {path} (backup: {bak})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
