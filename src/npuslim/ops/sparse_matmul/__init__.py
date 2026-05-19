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

import ctypes
import os

_LIB = None
_LIB_LOADED = False


def _load_lib():
    """Attempt to load libsparse_matmul.so. Idempotent."""
    global _LIB, _LIB_LOADED
    if _LIB_LOADED:
        return
    _LIB_LOADED = True

    so_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "libsparse_matmul.so"
    )
    if not os.path.isfile(so_path):
        return

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
    except OSError:
        pass


# Eagerly load at import time so Dynamo never hits filesystem ops.
_load_lib()


def _register_custom_op():
    """Register ``npu::sparse_matmul_4to2`` via torch.library."""
    import torch

    @torch.library.custom_op("npu::sparse_matmul_4to2", mutates_args=())
    def _sparse_matmul_4to2(
        a: torch.Tensor,
        b_dense: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        if a.ndim != 2 or b_dense.ndim != 2:
            raise ValueError(
                "sparse_matmul_4to2 expects 2D tensors: "
                f"got a.ndim={a.ndim}, b_dense.ndim={b_dense.ndim}"
            )

        m = a.shape[0]
        k = a.shape[1]
        n = b_dense.shape[0]
        if k % 4 != 0:
            raise ValueError(f"sparse_matmul_4to2 requires K % 4 == 0, got K={k}")
        if b_dense.shape[1] * 2 != k:
            raise ValueError(
                "sparse_matmul_4to2 expects b_dense.shape == [N, K//2], "
                f"got b_dense.shape={tuple(b_dense.shape)} for K={k}"
            )

        current_stream = torch.npu.current_stream(device=a.device)
        c = torch.zeros(m, n, dtype=torch.int32, device=a.device)

        ret = _LIB.run_sparse_matmul(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_dense.data_ptr()),
            ctypes.c_void_p(index.data_ptr()),
            ctypes.c_void_p(c.data_ptr()),
            ctypes.c_int64(m),
            ctypes.c_int64(n),
            ctypes.c_int64(k),
            current_stream._as_parameter_,
        )
        if ret != 0:
            raise RuntimeError(
                "sparse matmul kernel launch failed "
                f"(ret={ret}, M={m}, N={n}, K={k})"
            )
        return c

    @_sparse_matmul_4to2.register_fake
    def _(a, b_dense, index):
        return torch.empty(a.shape[0], b_dense.shape[0], dtype=torch.int32, device=a.device)


# Eagerly register if library is available.
if _LIB is not None:
    _register_custom_op()


def sparse_matmul_4to2(a, b_dense, index):
    """Compute C = A @ B_sparse via the AscendC 4:2 sparse matmul kernel.

    Args:
        a:       [M, K] int8 NPU tensor — left operand.
        b_dense: [N, K//2] int8 NPU tensor — densified 4:2 sparse B.
        index:   NZ-fractal format index tensor (uint8) on NPU.

    Returns:
        [M, N] int32 NPU tensor.
    """
    if _LIB is None:
        raise RuntimeError(
            "sparse_matmul_4to2 unavailable: libsparse_matmul.so not found"
        )
    import torch
    return torch.ops.npu.sparse_matmul_4to2(a, b_dense, index)
