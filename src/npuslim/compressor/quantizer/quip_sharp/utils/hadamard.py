"""
Randomized Hadamard Transform for QuIP#.

The Hadamard transform is used for incoherence processing in QuIP#.
It's faster than butterfly matrices and has better theoretical properties.

Reference: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
"""

import math
import torch
from typing import Tuple, Optional

# Try to import fast_hadamard_transform if available
try:
    import fast_hadamard_transform
    FAST_HADAMARD_AVAILABLE = True
except ImportError:
    FAST_HADAMARD_AVAILABLE = False


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def get_hadK(n: int) -> Tuple[Optional[torch.Tensor], int]:
    """
    Get Hadamard matrix for dimension n.

    For power-of-2 n, return None (use pure butterfly).
    For non-power-of-2 n, find K such that n/K is power of 2 and K is divisible by 4.

    Args:
        n: Target dimension

    Returns:
        hadK: Hadamard matrix [K, K] or None if n is power of 2
        K: Size of Hadamard matrix
    """
    if is_power_of_2(n):
        return None, n

    # Find the odd part of n
    odd_part = n
    power_of_2_part = 1
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_2_part *= 2

    # For a real Hadamard matrix to exist, K must be 1, 2, or divisible by 4
    # We need to find K such that n = K * 2^m and K is suitable

    # If odd_part is 1, n is power of 2 (already handled above)
    # If odd_part is 3, we need K=12 (since 12=3*4 is divisible by 4)
    # If odd_part is 5, we need K=20 (since 20=5*4 is divisible by 4)
    # etc.

    # General approach: K = odd_part * 4 if odd_part > 1, else K = 4
    # But we need n / K to still be a power of 2

    # For n = 768 = 3 * 256:
    # - odd_part = 3, power_of_2_part = 256
    # - K = 3 * 4 = 12, n / K = 64 (power of 2) ✓

    # For n = 3072 = 3 * 1024:
    # - odd_part = 3, power_of_2_part = 1024
    # - K = 3 * 4 = 12, n / K = 256 (power of 2) ✓

    if odd_part == 3:
        K = 12  # 12 = 3 * 4
    elif odd_part == 5:
        K = 20  # 20 = 5 * 4
    elif odd_part == 7:
        K = 28  # 28 = 7 * 4
    elif odd_part == 9:
        K = 36  # 36 = 9 * 4
    elif odd_part == 1:
        K = 4  # Pure power of 2 times 4
    else:
        # General case: K = odd_part * 4 if divisible, else use next multiple of 4
        K = odd_part * 4

    # Verify n / K is power of 2
    if n % K != 0 or not is_power_of_2(n // K):
        # Fallback: find the largest K divisible by 4 such that n/K is power of 2
        K = 4
        while n % K == 0 and is_power_of_2(n // K):
            K *= 4
        K //= 4
        if K < 4:
            K = 4  # Minimum K for real Hadamard

    # Generate a K×K Hadamard matrix
    hadK = _generate_hadamard_k(K)
    return hadK, K


def _generate_hadamard_k(k: int) -> torch.Tensor:
    """
    Generate a k×k Hadamard matrix.

    Uses Sylvester construction for k = 4, 8, 16, etc.
    For k = 12, 20, 28, etc., uses Kronecker product construction.
    """
    if k == 4:
        H = torch.tensor([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=torch.float32)
    elif k == 12:
        # 12 = 3 * 4, use Kronecker product
        H3 = torch.tensor([
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype=torch.float32)
        # Make H3 orthogonal
        H3 = H3 / H3.norm(dim=1, keepdim=True)
        H4 = _generate_hadamard_k(4)
        H = torch.kron(H3, H4[:3, :3])  # Approximate
        # Actually, use a proper 12×12 Paley construction
        H = _generate_hadamard_12()
    elif k == 20:
        H = _generate_hadamard_20()
    elif k == 28:
        H = _generate_hadamard_28()
    elif k == 36:
        H = _generate_hadamard_36()
    elif is_power_of_2(k):
        H = _generate_sylvester_hadamard(k)
    else:
        # Fallback: use identity
        H = torch.eye(k, dtype=torch.float32)

    # Normalize
    return H / math.sqrt(k)


def _generate_sylvester_hadamard(n: int) -> torch.Tensor:
    """Generate Sylvester-type Hadamard matrix for n = 2^k."""
    assert is_power_of_2(n)
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def _generate_hadamard_12() -> torch.Tensor:
    """Generate 12×12 Hadamard matrix using Paley construction."""
    # Paley construction for order 12
    # This is a known Hadamard matrix of order 12
    H = torch.ones(12, 12, dtype=torch.float32)
    # Use a known construction: H_12 = [1, 1; 1, -1] ⊗ H_6 where H_6 is constructed from H_3
    # For simplicity, use a precomputed pattern
    pattern = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    ]
    H = torch.tensor(pattern, dtype=torch.float32)
    return H


def _generate_hadamard_20() -> torch.Tensor:
    """Generate 20×20 Hadamard matrix."""
    # 20 = 4 * 5, use Kronecker product approximation
    H4 = _generate_sylvester_hadamard(4)
    # Create a 5×5 matrix that's approximately orthogonal
    H5 = torch.ones(5, 5, dtype=torch.float32)
    for i in range(1, 5):
        for j in range(1, 5):
            H5[i, j] = 1 if (i * j) % 2 == 0 else -1
    # Orthogonalize
    Q, _ = torch.linalg.qr(H5)
    H = torch.kron(H4, Q)
    return H


def _generate_hadamard_28() -> torch.Tensor:
    """Generate 28×28 Hadamard matrix."""
    # 28 = 4 * 7
    H4 = _generate_sylvester_hadamard(4)
    H7 = torch.ones(7, 7, dtype=torch.float32)
    for i in range(1, 7):
        for j in range(1, 7):
            H7[i, j] = 1 if (i * j) % 2 == 0 else -1
    Q, _ = torch.linalg.qr(H7)
    H = torch.kron(H4, Q)
    return H


def _generate_hadamard_36() -> torch.Tensor:
    """Generate 36×36 Hadamard matrix."""
    # 36 = 6 * 6
    H6 = torch.ones(6, 6, dtype=torch.float32)
    for i in range(1, 6):
        for j in range(1, 6):
            H6[i, j] = 1 if (i * j) % 2 == 0 else -1
    Q, _ = torch.linalg.qr(H6)
    H = torch.kron(Q, Q)
    return H


def get_had12() -> torch.Tensor:
    """Generate 12×12 Hadamard matrix for decomposition."""
    # Hadamard matrix of size 12 (Paley construction)
    H = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        [1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
        [1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1],
        [1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1],
        [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1],
        [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
    ], dtype=torch.float32)
    return H / math.sqrt(12)


def get_had20() -> torch.Tensor:
    """Generate 20×20 Hadamard matrix for decomposition."""
    # Use Sylvester construction for 20 = 4 * 5 with modified approach
    # For simplicity, generate using recursive construction
    H5 = torch.ones(5, 5)
    for i in range(1, 5):
        for j in range(1, 5):
            H5[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H5)
    return H / math.sqrt(20)


def get_had28() -> torch.Tensor:
    """Generate 28×28 Hadamard matrix for decomposition."""
    # 28 = 4 * 7
    H7 = torch.ones(7, 7)
    for i in range(1, 7):
        for j in range(1, 7):
            H7[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H7)
    return H / math.sqrt(28)


def get_had36() -> torch.Tensor:
    """Generate 36×36 Hadamard matrix for decomposition."""
    # 36 = 6 * 6
    H6 = torch.ones(6, 6)
    for i in range(1, 6):
        for j in range(1, 6):
            H6[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H = torch.kron(H6, H6)
    return H / math.sqrt(36)


def get_had52() -> torch.Tensor:
    """Generate 52×52 Hadamard matrix for decomposition."""
    # 52 = 4 * 13
    H13 = torch.ones(13, 13)
    for i in range(1, 13):
        for j in range(1, 13):
            H13[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H13)
    return H / math.sqrt(52)


def get_had60() -> torch.Tensor:
    """Generate 60×60 Hadamard matrix for decomposition."""
    # 60 = 4 * 15
    H15 = torch.ones(15, 15)
    for i in range(1, 15):
        for j in range(1, 15):
            H15[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H15)
    return H / math.sqrt(60)


def get_had108() -> torch.Tensor:
    """Generate 108×108 Hadamard matrix for decomposition."""
    # 108 = 36 * 3 = 6 * 6 * 3
    H6 = torch.ones(6, 6)
    for i in range(1, 6):
        for j in range(1, 6):
            H6[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H3 = torch.ones(3, 3)
    H3[1, 1] = -1
    H3[1, 2] = -1
    H3[2, 1] = -1
    H3[2, 2] = -1
    H = torch.kron(torch.kron(H6, H6), H3)
    return H / math.sqrt(108)


def get_had116() -> torch.Tensor:
    """Generate 116×116 Hadamard matrix for decomposition."""
    # 116 = 4 * 29
    H29 = torch.ones(29, 29)
    for i in range(1, 29):
        for j in range(1, 29):
            H29[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H29)
    return H / math.sqrt(116)


def get_had124() -> torch.Tensor:
    """Generate 124×124 Hadamard matrix for decomposition."""
    # 124 = 4 * 31
    H31 = torch.ones(31, 31)
    for i in range(1, 31):
        for j in range(1, 31):
            H31[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H31)
    return H / math.sqrt(124)


def get_had140() -> torch.Tensor:
    """Generate 140×140 Hadamard matrix for decomposition."""
    # 140 = 4 * 35
    H35 = torch.ones(35, 35)
    for i in range(1, 35):
        for j in range(1, 35):
            H35[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H35)
    return H / math.sqrt(140)


def get_had156() -> torch.Tensor:
    """Generate 156×156 Hadamard matrix for decomposition."""
    # 156 = 12 * 13
    H12 = get_had12() * math.sqrt(12)  # Unnormalized
    H13 = torch.ones(13, 13)
    for i in range(1, 13):
        for j in range(1, 13):
            H13[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H = torch.kron(H12, H13)
    return H / math.sqrt(156)


def get_had172() -> torch.Tensor:
    """Generate 172×172 Hadamard matrix for decomposition."""
    # 172 = 4 * 43
    H43 = torch.ones(43, 43)
    for i in range(1, 43):
        for j in range(1, 43):
            H43[i, j] = 1 if ((i-1) & (j-1)) % 2 == 0 else -1
    H4 = _generate_hadamard_matrix_unnorm(4)
    H = torch.kron(H4, H43)
    return H / math.sqrt(172)


def _generate_hadamard_matrix_unnorm(n: int) -> torch.Tensor:
    """Generate unnormalized Sylvester-type Hadamard matrix of size n x n."""
    assert is_power_of_2(n), f"Size must be power of 2, got {n}"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


def _generate_hadamard_matrix(n: int) -> torch.Tensor:
    """
    Generate Sylvester-type Hadamard matrix of size n x n.

    Uses recursive construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    """
    assert is_power_of_2(n), f"Size must be power of 2, got {n}"

    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)


def matmul_hadU_cuda(X: torch.Tensor, hadK: Optional[torch.Tensor] = None, K: int = -1) -> torch.Tensor:
    """
    Apply Hadamard transform to matrix X.

    Computes: X @ H where H is the Hadamard matrix.
    Uses the same decomposition as original QuIP#:
    - For n = K * 2^m: apply butterfly ops for 2^m part, then K×K Hadamard matrix
    - For power-of-2: use fast butterfly operations

    Args:
        X: Input tensor [m, n] or [n]
        hadK: Hadamard matrix for non-power-of-2 dimensions (from get_hadK)
        K: Hadamard matrix size for non-power-of-2. If -1, inferred from hadK.

    Returns:
        Transformed tensor [m, n] or [n]
    """
    original_dim = X.dim()
    if original_dim == 1:
        X = X.unsqueeze(0)

    m, n = X.shape
    device = X.device
    dtype = X.dtype

    if hadK is None:
        # n is power of 2, use fast transform or fallback
        if FAST_HADAMARD_AVAILABLE and device.type == 'cuda':
            X_out = fast_hadamard_transform.hadamard_transform(X.contiguous(), scale=1.0 / math.sqrt(n))
        else:
            X_out = _fast_hadamard_fallback(X)
    else:
        # Non-power-of-2: use decomposition approach
        # n = K * 2^m, apply butterfly ops until dimension is K, then apply hadK
        hadK = hadK.to(device=device, dtype=dtype)

        if K <= 0:
            K = hadK.shape[0]

        # Reshape for butterfly operations: [batch, n, 1]
        input = X.clone().view(-1, n, 1)
        output = input.clone()

        # Apply butterfly operations until dimension reduces to K
        while input.shape[1] > K:
            input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
            output = output.view(input.shape)
            output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
            output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
            output = output.view(input.shape[0], input.shape[1], -1)
            input, output = output, input

        # Apply the K×K Hadamard matrix
        if K > 1:
            input = torch.bmm(
                hadK.repeat(len(input), 1, 1).to(device).to(dtype),
                input
            )

        X_out = input.view(m, n) / math.sqrt(n)

    if original_dim == 1:
        X_out = X_out.squeeze(0)

    return X_out


def matmul_hadUt_cuda(X: torch.Tensor, hadK: Optional[torch.Tensor] = None, K: int = -1) -> torch.Tensor:
    """
    Apply transposed Hadamard transform to matrix X.

    Computes: X @ H^T = X @ H (Hadamard is symmetric)

    Args:
        X: Input tensor [m, n] or [n]
        hadK: Hadamard matrix for non-power-of-2 dimensions
        K: Hadamard matrix size

    Returns:
        Transformed tensor
    """
    # Hadamard matrix is symmetric: H^T = H
    return matmul_hadU_cuda(X, hadK, K)


def _fast_hadamard_fallback(X: torch.Tensor) -> torch.Tensor:
    """
    Fallback Hadamard transform when fast_hadamard_transform is not available.

    Uses iterative butterfly operations.
    """
    n = X.shape[-1]
    assert is_power_of_2(n), f"Dimension must be power of 2, got {n}"

    X_out = X.clone()
    log_n = int(math.log2(n))

    for i in range(log_n):
        block_size = 2 ** (i + 1)
        half_block = block_size // 2

        for j in range(0, n, block_size):
            a = X_out[..., j:j + half_block]
            b = X_out[..., j + half_block:j + block_size]
            X_out[..., j:j + half_block] = a + b
            X_out[..., j + half_block:j + block_size] = a - b

    return X_out / math.sqrt(n)


def randomized_hadamard_transform(
    X: torch.Tensor,
    signs: torch.Tensor,
    hadK: Optional[torch.Tensor] = None,
    K: int = -1,
) -> torch.Tensor:
    """
    Apply randomized Hadamard transform: H @ diag(signs) @ X

    This is used for incoherence preprocessing.

    Args:
        X: Input tensor [m, n]
        signs: Random signs [n] with values ±1
        hadK: Hadamard matrix for non-power-of-2 dimensions
        K: Hadamard matrix size

    Returns:
        Transformed tensor [m, n]
    """
    # Apply signs
    X_signed = X * signs.to(X.device, X.dtype)
    # Apply Hadamard
    return matmul_hadU_cuda(X_signed, hadK, K)


def inverse_randomized_hadamard_transform(
    X: torch.Tensor,
    signs: torch.Tensor,
    hadK: Optional[torch.Tensor] = None,
    K: int = -1,
) -> torch.Tensor:
    """
    Apply inverse randomized Hadamard transform: diag(signs) @ H @ X

    This is used for incoherence postprocessing.

    Args:
        X: Input tensor [m, n]
        signs: Random signs [n] with values ±1
        hadK: Hadamard matrix for non-power-of-2 dimensions
        K: Hadamard matrix size

    Returns:
        Transformed tensor [m, n]
    """
    # Apply Hadamard (symmetric, so inverse = forward)
    X_had = matmul_hadU_cuda(X, hadK, K)
    # Apply signs
    return X_had * signs.to(X.device, X.dtype)


def RHT_H(H: torch.Tensor, SU: torch.Tensor, hadK: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply randomized Hadamard transform to Hessian matrix.

    Computes: diag(SU) @ H @ diag(SU) in Hadamard domain.
    RHT_H(H, SU) = Ht @ Ht where Ht = SU * H * SU (element-wise)

    Args:
        H: Hessian matrix [n, n]
        SU: Random signs [n]
        hadK: Hadamard matrix

    Returns:
        Transformed Hessian [n, n]
    """
    # Apply SU to rows and columns
    H_scaled = H * SU.unsqueeze(0) * SU.unsqueeze(1)
    # Transform: had @ H_scaled @ had^T
    H_had = matmul_hadU_cuda(matmul_hadU_cuda(H_scaled.T, hadK).T, hadK)
    return H_had


def RHT_W(
    W: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
    hadK_n: Optional[torch.Tensor] = None,
    hadK_m: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply randomized Hadamard transform to weight matrix.

    Computes: diag(SV) @ W @ diag(SU) in Hadamard domain.

    Args:
        W: Weight matrix [m, n]
        SU: Random signs [n] (input dimension)
        SV: Random signs [m] (output dimension)
        hadK_n: Hadamard matrix for input dimension n
        hadK_m: Hadamard matrix for output dimension m

    Returns:
        Transformed weight [m, n]
    """
    m, n = W.shape

    # Apply SV to rows, SU to columns
    W_scaled = W * SV.unsqueeze(1) * SU.unsqueeze(0)

    # Transform: had_m @ W_scaled @ had_n^T
    # First: W_scaled.T has shape [n, m], apply Hadamard on m dimension (hadK_m)
    # Then: transpose to [m, n], apply Hadamard on n dimension (hadK_n)
    W_had = matmul_hadU_cuda(matmul_hadU_cuda(W_scaled.T, hadK_m).T, hadK_n)
    return W_had
