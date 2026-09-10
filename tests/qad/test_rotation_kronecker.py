"""create_rot 非 2 幂 Kronecker 构造单测（GLM-5.2 hidden=6144=12×512）。

覆盖：pow2 路径无回归、非 pow2 Kronecker 正交性、与 H_12 ⊗ H_512 结构一致。
"""

import math

import torch

from npuslim.qad.rotation import (
    _matmul_had_u,
    _matmul_had_u_k,
    load_hadamard_matrix_from_txt,
    random_hadamard_matrix,
)


def test_pow2_path_regression():
    torch.manual_seed(2026)
    H = random_hadamard_matrix(256)
    err = (H @ H.t() - torch.eye(256)).abs().max().item()
    assert err < 1e-4


def test_kronecker_orthogonal_6144():
    torch.manual_seed(2026)
    H = random_hadamard_matrix(6144)  # 12 * 512, non-pow2 -> Kronecker path
    assert H.shape == (6144, 6144)
    err = (H @ H.t() - torch.eye(6144)).abs().max().item()
    assert err < 1e-3


def test_kronecker_matches_h12_kron_h512():
    n, k, m = 6144, 12, 512

    def walsh_raw(size):
        H = torch.eye(size)
        return _matmul_had_u(H, size) * math.sqrt(size)

    H512 = walsh_raw(m)
    H12 = load_hadamard_matrix_from_txt("had.12.txt")
    H_kron = torch.kron(H12, H512)

    H_n = torch.zeros(n, n)
    for j in range(n):
        e = torch.zeros(n)
        e[j] = 1
        H_n[:, j] = _matmul_had_u_k(e, H12, k, n)
    H_n_un = H_n * math.sqrt(n)
    err = (H_n_un - H_kron).abs().max().item()
    assert err < 1e-2


def test_create_rot_hadamard_6144_ok():
    from npuslim.qad.rotation import QuaRotMode, create_rot

    r1 = create_rot(QuaRotMode.HADAMARD, 6144, block_size=-1, seed=2026)
    assert r1.shape == (6144, 6144)
    err = (r1 @ r1.t() - torch.eye(6144)).abs().max().item()
    assert err < 1e-3
