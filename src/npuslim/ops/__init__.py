"""Custom operator packages for npuslim.

Each sub-package corresponds to a hardware-accelerated operator
(e.g., sparse_matmul). Operators are loaded lazily — if the
underlying .so is not available, the module still imports but
calls will raise RuntimeError.
"""
