#!/usr/bin/env python3
"""Grade ONE LiveCodeBench problem in an isolated subprocess (segfault containment).

stdin : JSON {"sample": {...}, "code": "..."}
stdout: JSON {"res": [...], "meta": {...}}
A candidate that segfaults (e.g. deep recursion blowing the C stack) kills only
this child; the parent records the problem as failed (-6).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcb_testing_util import run_test  # noqa: E402


def main():
    d = json.loads(sys.stdin.read())
    try:
        res, meta = run_test(d["sample"], test=d["code"], debug=False, timeout=6)
        fixed = [bool(x) if isinstance(x, bool) else int(x) for x in res]
        print(json.dumps({"res": fixed, "meta": {"ok": True}}))
    except Exception as e:
        print(json.dumps({"res": [-5], "meta": {"error": repr(e)[:200]}}))


if __name__ == "__main__":
    main()
