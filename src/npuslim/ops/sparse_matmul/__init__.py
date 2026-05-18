"""AscendC 4:2 structured sparse matmul operator.

Usage::

    from npuslim.ops.sparse_matmul import sparse_matmul_4to2

    # a:       [M, K] int8 NPU tensor
    # b_dense: [N, K//2] int8 NPU tensor (densified 4:2 sparse B)
    # index:   NZ-fractal index tensor, uint8, on NPU
    c = sparse_matmul_4to2(a, b_dense, index)   # -> [M, N] int32 NPU tensor

The caller is responsible for 4:2 sparse compression of B
(producing b_dense and index). This operator only runs the
hardware-accelerated sparse matmul kernel.
"""

import os
import ctypes

_LIB = None
_LIB_LOADED = False
_torch = None


def _ensure_lib():
    """Load libsparse_matmul.so and import torch. Returns True on success."""
    global _LIB, _LIB_LOADED, _torch
    if _LIB_LOADED:
        return _LIB is not None

    _LIB_LOADED = True

    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsparse_matmul.so")
    if not os.path.isfile(so_path):
        return False

    try:
        import torch
        _torch = torch
    except ImportError:
        return False

    try:
        _LIB = ctypes.CDLL(so_path)
        _LIB.run_sparse_matmul.restype = ctypes.c_int
        _LIB.run_sparse_matmul.argtypes = [
            ctypes.c_void_p,  # a_dev
            ctypes.c_void_p,  # b_dev
            ctypes.c_void_p,  # index_dev
            ctypes.c_void_p,  # c_dev
            ctypes.c_int64,   # m
            ctypes.c_int64,   # n
            ctypes.c_int64,   # k
            ctypes.c_void_p,  # stream
        ]
        return True
    except OSError:
        return False


_op_registered = False


def _register_custom_op():
    """Register ``npu::sparse_matmul_4to2`` via torch.library."""
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    import torch

    @torch.library.custom_op("npu::sparse_matmul_4to2", mutates_args=())
    def _sparse_matmul_4to2(
        a: torch.Tensor,
        b_dense: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        m = a.shape[0]
        k = a.shape[1]
        n = b_dense.shape[0]

        c = torch.zeros(m, n, dtype=torch.int32, device=a.device)

        ret = _LIB.run_sparse_matmul(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_dense.data_ptr()),
            ctypes.c_void_p(index.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_int64(m),
            ctypes.c_int64(n),
            ctypes.c_int64(k),
            ctypes.c_void_p(0),
        )
        if ret != 0:
            raise RuntimeError(f"sparse matmul kernel launch failed (ret={ret})")
        return c


def sparse_matmul_4to2(a, b_dense, index):
    """Compute C = A @ B_sparse via the AscendC 4:2 sparse matmul kernel.

    Args:
        a:       [M, K] int8 NPU tensor — left operand.
        b_dense: [N, K//2] int8 NPU tensor — densified 4:2 sparse B.
        index:   NZ-fractal format index tensor (uint8) on NPU.

    Returns:
        [M, N] int32 NPU tensor.
    """
    if not _ensure_lib():
        raise RuntimeError(
            "sparse_matmul_4to2 unavailable: libsparse_matmul.so not found "
            "or torch not installed"
        )
    _register_custom_op()
    # Direct function call (also accessible as torch.ops.npu.sparse_matmul_4to2)
    return _torch.ops.npu.sparse_matmul_4to2(a, b_dense, index)
