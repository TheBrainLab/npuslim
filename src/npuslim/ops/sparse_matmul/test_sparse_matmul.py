#!/usr/bin/env python3
"""Test for the sparse_matmul operator.

Usage:
    python -m npuslim.ops.sparse_matmul.test_sparse_matmul
"""

import time
import numpy as np

import torch
import torch_npu

from npuslim.ops.sparse_matmul import sparse_matmul_4to2


# ---- 4:2 sparse test utilities ----

def _gen_sparse_matrix_b(n, k, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    b = np.zeros((n, k), dtype=np.int8)
    for row in range(n):
        for i in range(0, k, 4):
            nz_pos = rng.choice(4, 2, replace=False)
            for p in nz_pos:
                b[row, i + p] = rng.integers(-5, 6, dtype=np.int8)
                if b[row, i + p] == 0:
                    b[row, i + p] = 1
    return b


def _densify_and_gen_index(b_matrix):
    n, k = b_matrix.shape
    b_dense = np.zeros((n, k // 2), dtype=b_matrix.dtype)
    index_matrix = np.zeros((n, k // 8), dtype=np.uint8)
    index_mask = np.zeros((n, k // 2), dtype=np.uint32)

    for row in range(n):
        dense_row, index_row, index_mask_row = [], [], []
        for i in range(0, k, 4):
            block = b_matrix[row, i:i + 4]
            nz_pos = [j for j in range(4) if block[j] != 0]

            if len(nz_pos) == 0:
                idx1, idx2 = 0, 0
                index_mask_row.extend([i, i])
            elif len(nz_pos) == 1:
                idx1 = nz_pos[0] if nz_pos[0] < 3 else 0
                idx2 = 0 if nz_pos[0] < 3 else 2
                index_mask_row.extend([nz_pos[0] + i, i])
            else:
                idx1 = nz_pos[0]
                idx2 = nz_pos[1] - 1
                index_mask_row.extend([nz_pos[0] + i, nz_pos[1] + i])

            index_row.extend([idx1, idx2])
            dense_block = [block[pos] for pos in nz_pos[:2]]
            if len(dense_block) < 2:
                dense_block += [0] * (2 - len(dense_block))
            dense_row.extend(dense_block)

        index_bytes = []
        for j in range(0, len(index_row), 4):
            indices = index_row[j:j + 4]
            index_bytes.append(sum(idx << (2 * bit) for bit, idx in enumerate(indices)))

        b_dense[row, :] = dense_row
        index_matrix[row, :] = index_bytes
        index_mask[row, :] = index_mask_row

    return b_dense, index_matrix, index_mask


def _index_nd_to_tiled(index_matrix, k_tile_index=8):
    """Convert ND index to the tiled format expected by the AscendC sparse library.

    The AscendC CopySparseIdxToCubeFromGM reads the index with the layout:
        srcOffset = row * c0Size + col * alignedGRow
    where c0Size=8 and alignedGRow = ceil(N/16)*16.

    This means each K_tile's index data must be stored as a contiguous
    [N_padded, k_tile_index] block, with blocks concatenated sequentially.
    """
    n, k = index_matrix.shape
    pad_n = int(np.ceil(n / 16) * 16)
    padded = np.zeros((pad_n, k), dtype=np.uint8)
    padded[:n, :k] = index_matrix
    chunks = []
    for i in range(0, k, k_tile_index):
        chunk = padded[:, i:i + k_tile_index]
        if chunk.shape[1] < k_tile_index:
            chunk = np.pad(chunk, ((0, 0), (0, k_tile_index - chunk.shape[1])))
        chunks.append(chunk.flatten())
    return np.concatenate(chunks)


def main():
    np.random.seed(42)
    M, N, K = 128, 768, 128

    a_np = np.random.randint(-5, 5, (M, K), dtype=np.int8)
    b_np = _gen_sparse_matrix_b(N, K)

    # Golden: dense matmul with zeros preserved
    c_golden = np.dot(a_np.astype(np.int32), b_np.T.astype(np.int32)).astype(np.int32)

    # Compress B
    b_dense_np, index_np, _ = _densify_and_gen_index(b_np)
    index_tiled_np = _index_nd_to_tiled(index_np)

    # Move to NPU
    a_t = torch.from_numpy(a_np).npu()
    b_dense_t = torch.from_numpy(b_dense_np).npu()
    index_t = torch.from_numpy(index_tiled_np).npu()

    # --- Correctness (first run also serves as warm-up) ---
    c_t = sparse_matmul_4to2(a_t, b_dense_t, index_t)
    torch.npu.synchronize()
    c_hw = c_t.cpu().numpy()

    diff = np.abs(c_hw.astype(np.int64) - c_golden.astype(np.int64))
    max_diff = diff.max()
    wrong = int((diff > 0).sum())
    total = M * N

    print(f"Shape: A=[{M},{K}], B=[{N},{K}], C=[{M},{N}]")
    print(f"max_diff = {max_diff}, wrong/total = {wrong}/{total}")
    print(f"result   = {'PASS' if wrong == 0 else 'FAIL'}")
    assert wrong == 0, f"{wrong} mismatches"

    # --- Latency benchmark (exclude cold-start) ---
    WARMUP = 3
    ROUNDS = 10

    # Sparse matmul
    for _ in range(WARMUP):
        sparse_matmul_4to2(a_t, b_dense_t, index_t)
        torch.npu.synchronize()

    sparse_lat = []
    for _ in range(ROUNDS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        sparse_matmul_4to2(a_t, b_dense_t, index_t)
        torch.npu.synchronize()
        sparse_lat.append(time.perf_counter() - t0)

    # Dense matmul baseline: A [M,K] int8 @ B_full [K,N] int8^T -> [M,N]
    # torch mm on NPU casts to float, so we compare end-to-end wall time.
    b_full_t = torch.from_numpy(b_np).npu()
    a_f = a_t.float()
    b_f = b_full_t.float()

    for _ in range(WARMUP):
        _ = torch.mm(a_f, b_f.T)
        torch.npu.synchronize()

    dense_lat = []
    for _ in range(ROUNDS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = torch.mm(a_f, b_f.T)
        torch.npu.synchronize()
        dense_lat.append(time.perf_counter() - t0)

    def _stats(secs):
        us = [t * 1e6 for t in secs]
        return sum(us) / len(us), min(us), max(us)

    s_avg, s_lo, s_hi = _stats(sparse_lat)
    d_avg, d_lo, d_hi = _stats(dense_lat)

    print(f"\nLatency ({ROUNDS} rounds, {WARMUP} warm-up excluded):")
    print(f"  sparse (int8 4:2 -> int32) : avg = {s_avg:>8.1f} us, min = {s_lo:.1f}, max = {s_hi:.1f}")
    print(f"  dense  (fp32 mm)           : avg = {d_avg:>8.1f} us, min = {d_lo:.1f}, max = {d_hi:.1f}")
    print(f"  speedup (dense / sparse)   : {d_avg / s_avg:.2f}x")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
