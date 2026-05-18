"""Shared platform detection for npuslim operator builds."""

import os
import subprocess
import sys


def is_npu_available():
    """Check if NPU/CANN environment is present."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    return bool(ascend_home) and os.path.isdir(ascend_home)


def is_gpu_available():
    """Check if CUDA environment is present."""
    cuda_home = os.environ.get("CUDA_HOME", "") or os.environ.get("CUDA_PATH", "")
    if cuda_home and os.path.isdir(cuda_home):
        return True
    # Fallback: check for nvcc or nvidia-smi
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def detect_soc_version():
    """Detect Ascend SoC version via acl.get_soc_name().

    Returns the SoC string (e.g. "Ascend910_9392") or None.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import acl; print(acl.get_soc_name())"],
            capture_output=True, text=True, timeout=10,
        )
        soc = result.stdout.strip()
        if result.returncode == 0 and soc:
            return soc
    except Exception:
        pass
    return None


def get_ascend_home():
    """Return ASCEND_HOME_PATH or None."""
    path = os.environ.get("ASCEND_HOME_PATH", "")
    return path if path and os.path.isdir(path) else None
