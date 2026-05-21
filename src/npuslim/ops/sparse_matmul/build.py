"""Build script for the sparse_matmul AscendC operator.

Platform: NPU only.

Usage:
    python -m npuslim.ops.sparse_matmul.build
"""

import os
import shutil
import subprocess
import sys
import warnings

# _platform.py only uses stdlib — import directly without triggering npuslim.__init__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _platform import get_ascend_home, detect_soc_version  # noqa: E402

PLATFORM = "npu"


def build(verbose=False):
    """Build libsparse_matmul.so. Returns True on success."""
    op_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(op_dir, "csrc")
    build_dir = os.path.join(csrc_dir, "build")
    so_src = os.path.join(build_dir, "libsparse_matmul.so")
    so_dst = os.path.join(op_dir, "libsparse_matmul.so")

    # --- Platform check ---
    ascend_home = get_ascend_home()
    if not ascend_home:
        warnings.warn(
            "sparse_matmul: ASCEND_HOME_PATH not set — skipping NPU operator build.",
            stacklevel=2,
        )
        return False

    soc_version = detect_soc_version()
    if not soc_version:
        warnings.warn(
            "sparse_matmul: cannot detect SoC version — skipping build.",
            stacklevel=2,
        )
        return False

    print(f"[sparse_matmul] Building for {soc_version} (CANN: {ascend_home})")

    # --- CMake configure + build ---
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    cmake_cmd = [
        "cmake",
        "-DRUN_MODE=npu",
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_home}",
        "..",
    ]
    make_cmd = ["make", f"-j{os.cpu_count() or 4}"]

    try:
        subprocess.run(cmake_cmd, cwd=build_dir, check=True,
                       capture_output=not verbose)
        subprocess.run(make_cmd, cwd=build_dir, check=True,
                       capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"sparse_matmul: build failed (exit {e.returncode}).",
            stacklevel=2,
        )
        return False

    # --- Copy .so to package directory ---
    if not os.path.isfile(so_src):
        warnings.warn(
            "sparse_matmul: build succeeded but .so not found.", stacklevel=2,
        )
        return False

    shutil.copy2(so_src, so_dst)
    print(f"[sparse_matmul] Installed: {so_dst}")
    return True


if __name__ == "__main__":
    ok = build(verbose=True)
    sys.exit(0 if ok else 1)
