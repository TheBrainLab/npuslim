"""
LDLQ (Low-rank Decomposition Lattice Quantization) algorithms from QuIP.

This module provides the core LDLQ rounding algorithms used by QuIP for
weight quantization with Hessian-based error minimization.

Reference: https://github.com/Cornell-RelaxML/QuIP/blob/master/vector_balance.py
"""

from dataclasses import dataclass
from typing import Optional, Literal
import sys
import torch


@dataclass
class LDLQConfig:
    """Configuration for LDLQ quantization algorithm."""

    # Basic quantization
    nbits: int = 4
    quant_func: Literal["minmax", "rms"] = "rms"  # Quantization function type

    # LDLQ algorithm settings
    ldlq_method: Literal["ldlq", "ldlqRG", "allbal", "ldlbal_admm", "ldl_gptqequiv"] = "ldlq"
    npasses: int = 0  # Number of greedy refinement passes
    unbiased: bool = False  # Use unbiased rounding
    blocksize: int = 128  # Block size for blocked variants

    # For minmax mode (pre-computed scale/zero)
    scale: Optional[torch.Tensor] = None
    zero: Optional[torch.Tensor] = None


def check_nbits(wr, nbits):
    """Check that quantized values don't exceed nbits range."""
    wr_vals, wr_counts = torch.unique(wr, sorted=True, return_counts=True)
    assert len(wr_vals) <= 2**nbits
    return wr_counts


def _allonce(x, w, unbiased=False):
    """Round to nearest integer with optional unbiased rounding."""
    if unbiased:
        z = torch.floor(w - x + torch.rand(x.shape).to(x.device))
    else:
        z = torch.round(w - x)
    return w - z


# ============================================================
# LDLQ Rounding Functions
# ============================================================


def round_ldl(w, H, nbits, n_greedy_passes=9, unbiased=False):
    """
    Standard LDL rounding algorithm.

    Args:
        w: Weight tensor [m, d] in [0, maxq] range
        H: Hessian matrix [d, d]
        nbits: Number of bits
        n_greedy_passes: Number of greedy refinement passes
        unbiased: Use unbiased rounding

    Returns:
        Quantized weight tensor
    """
    assert (not unbiased) or (n_greedy_passes == 0), \
        "greedy passes are incompatible with unbiased LDL rounding"

    d, d_ = H.shape
    assert d == d_
    m, d = w.shape

    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)

    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)

    w_hat = w.clone()
    for i in reversed(range(d)):
        w_hat[:, i] = torch.clamp(
            torch.floor(w[:, i] + (w[:, i:] - w_hat[:, i:]) @ L[i:, i] + eta[:, i]),
            min=0,
            max=2**nbits - 1,
        )

    wr = w_hat.clone()
    s = w_hat - w
    H = H / H.diag().max()

    for igp in range(n_greedy_passes):
        for i in reversed(range(d)):
            Hs = s @ H[:, i]
            epsXTsj = wr[:, i] - torch.round(wr[:, i] - Hs / H[i, i])
            wr[:, i] -= epsXTsj
            s[:, i] -= epsXTsj
        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if (w_hat == wr).all():
            sys.stderr.write(f"breaking after {igp+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)

    check_nbits(wr, nbits)
    return wr


def round_ldl_block(w, H, nbits, blocksize=128, n_greedy_passes=9, unbiased=False):
    """Blocked version of LDL rounding for memory efficiency."""
    assert (not unbiased) or (n_greedy_passes == 0), \
        "greedy passes are incompatible with unbiased LDL rounding"

    d, d_ = H.shape
    assert d == d_
    m, d = w.shape

    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)

    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)

    w_hat = w.clone()
    for i2 in range(d, 0, -blocksize):
        i1 = max(i2 - blocksize, 0)
        count = i2 - i1
        W1 = w[:, i1:i2]
        W2Hdiff = w[:, i2:] - w_hat[:, i2:]
        WHat1 = w_hat[:, i1:i2].clone()
        L1 = L[:, i1:i2]
        Eta1 = eta[:, i1:i2]

        for i in reversed(range(count)):
            WHat1[:, i] = torch.clamp(
                torch.floor(
                    W1[:, i]
                    + (W1 - WHat1) @ L1[i1:i2, i]
                    + W2Hdiff @ L1[i2:, i]
                    + Eta1[:, i]
                ),
                min=0,
                max=2**nbits - 1,
            )

        w_hat[:, i1:i2] = WHat1

    wr = w_hat.clone()
    s = w_hat - w
    H = H / H.diag().max()

    for igp in range(n_greedy_passes):
        for i2 in range(d, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            W1 = wr[:, i1:i2].clone()
            S0 = s[:, :i1]
            S1 = s[:, i1:i2].clone()
            S2 = s[:, i2:]
            H0 = H[:i1, i1:i2]
            H1 = H[i1:i2, i1:i2]
            H2 = H[i2:, i1:i2]

            for i in reversed(range(count)):
                Hs = S0 @ H0[:, i] + S1 @ H1[:, i] + S2 @ H2[:, i]
                epsXTsj = W1[:, i] - torch.round(W1[:, i] - Hs / H1[i, i])
                W1[:, i] -= epsXTsj
                S1[:, i] -= epsXTsj

            wr[:, i1:i2] = W1
            s[:, i1:i2] = S1

        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if (w_hat == wr).all():
            sys.stderr.write(f"breaking after {igp+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)

    check_nbits(wr, nbits)
    return wr


def round_sorted_ldlqRG(w, H, nbits, n_greedy_passes=9, unbiased=False, pivot=None):
    """LDL with Hessian diagonal sorting (reverse greedy)."""
    p = torch.argsort(torch.diag(H))
    Hp = H[p, :][:, p]
    wp = w[:, p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:, p] = round_ldl(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr


def round_sorted_ldlqRG_block(w, H, nbits, blocksize=128, n_greedy_passes=9, unbiased=False, pivot=None):
    """Blocked LDL with Hessian diagonal sorting."""
    p = torch.argsort(torch.diag(H))
    Hp = H[p, :][:, p]
    wp = w[:, p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:, p] = round_ldl_block(wp, Hp, nbits, blocksize, n_greedy_passes, unbiased)
    return wr


def round_allbal(w, H, nbits, npasses, unbiased=False, calc_entropy=False):
    """All-balance rounding (vector balance without LDL)."""
    d, d_ = H.shape
    assert d == d_
    m, d = w.shape

    wr = w.clone()
    s = torch.zeros(m, d).to(w.device)
    w_hat = wr.clone()

    H = H / H.diag().max()

    for ip in range(npasses):
        for i in range(d):
            Hs = s @ H[:, i]
            epsXTsj = _allonce(Hs / H[i, i], wr[:, i], unbiased=unbiased)
            wr[:, i] -= epsXTsj
            s[:, i] -= epsXTsj
        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if (w_hat == wr).all():
            sys.stderr.write(f"breaking after {ip+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)

    check_nbits(wr, nbits)
    return wr


def round_allbal_block(w, H, nbits, npasses, blocksize=128, unbiased=False, calc_entropy=False):
    """Blocked all-balance rounding."""
    d, d_ = H.shape
    assert d == d_
    m, d = w.shape

    wr = w.clone()
    s = torch.zeros(m, d).to(w.device)
    w_hat = wr.clone()

    H = H / H.diag().max()

    for ip in range(npasses):
        for i1 in range(0, d, blocksize):
            i2 = min(i1 + blocksize, d)
            count = i2 - i1
            W1 = wr[:, i1:i2].clone()
            S0 = s[:, :i1]
            S1 = s[:, i1:i2].clone()
            S2 = s[:, i2:]
            H0 = H[:i1, i1:i2]
            H1 = H[i1:i2, i1:i2]
            H2 = H[i2:, i1:i2]

            for i in range(count):
                Hs = S0 @ H0[:, i] + S1 @ H1[:, i] + S2 @ H2[:, i]
                epsXTsj = _allonce(Hs / H1[i, i], W1[:, i], unbiased=unbiased)
                W1[:, i] -= epsXTsj
                S1[:, i] -= epsXTsj

            wr[:, i1:i2] = W1
            s[:, i1:i2] = S1

        wr = torch.clamp(wr, min=0, max=2**nbits - 1)
        if (w_hat == wr).all():
            sys.stderr.write(f"breaking after {ip+1} greedy passes found fixed point")
            break
        w_hat.copy_(wr)

    check_nbits(wr, nbits)
    return wr


def round_sorted_ldl_admm(w, H, nbits, n_greedy_passes=9, unbiased=False, pivot=None):
    """LDL with ADMM optimization."""
    # Simplified implementation - just use standard LDL
    # Full ADMM implementation can be added if needed
    p = torch.argsort(torch.diag(H))
    Hp = H[p, :][:, p]
    wp = w[:, p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:, p] = round_ldl(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr


def round_ldl_gptqequiv(w, H, nbits, unbiased=False):
    """LDL equivalent to GPTQ's algorithm (forward order)."""
    d, d_ = H.shape
    assert d == d_
    m, d = w.shape

    H = torch.flip(H, [0, 1])
    L = torch.linalg.cholesky(H)
    L = torch.flip(L, [0, 1])
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)

    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)

    w_hat = w.clone()
    for i in range(d):
        w_hat[:, i] = torch.clamp(
            torch.floor(
                w[:, i]
                + (w[:, : i + 1] - w_hat[:, : i + 1]) @ L[: i + 1, i]
                + eta[:, i]
            ),
            min=0,
            max=2**nbits - 1,
        )

    check_nbits(w_hat, nbits)
    return w_hat


# ============================================================
# Main Quantization Function
# ============================================================


def round_ldlq(
    w: torch.Tensor,
    H: torch.Tensor,
    config: LDLQConfig,
) -> torch.Tensor:
    """
    Apply LDLQ rounding to normalized weights.

    This is the core rounding function that dispatches to specific
    LDLQ variants based on config.ldlq_method.

    Args:
        w: Normalized weight tensor [m, d] in [0, maxq] range
        H: Hessian matrix [d, d]
        config: LDLQ configuration

    Returns:
        Quantized weight tensor [m, d] in [0, maxq] range
    """
    # Method registry: (standard_function, block_function)
    method_registry = {
        "ldlq": (round_ldl, round_ldl_block),
        "ldlqRG": (round_sorted_ldlqRG, round_sorted_ldlqRG_block),
        "allbal": (round_allbal, round_allbal_block),
    }

    # Handle standalone methods
    if config.ldlq_method == "ldlbal_admm":
        return round_sorted_ldl_admm(
            w, H,
            nbits=config.nbits,
            n_greedy_passes=config.npasses,
            unbiased=config.unbiased,
        )
    if config.ldlq_method == "ldl_gptqequiv":
        return round_ldl_gptqequiv(
            w, H,
            nbits=config.nbits,
            unbiased=config.unbiased,
        )

    # Handle registry methods
    if config.ldlq_method in method_registry:
        std_fn, block_fn = method_registry[config.ldlq_method]
        # Use blocked version for large matrices
        target_fn = block_fn if w.shape[1] > config.blocksize else std_fn

        # allbal requires sorting by Hessian diagonal
        if config.ldlq_method == "allbal":
            Hdiag = H.diag()
            p = Hdiag.sort(descending=True).indices
            Hp = H[:, p][p, :]
            wp = w[:, p]

            wp_hat = target_fn(
                wp, Hp,
                nbits=config.nbits,
                npasses=config.npasses,
                unbiased=config.unbiased,
                blocksize=config.blocksize,
            )

            # Re-invert order
            ip = torch.argsort(p)
            return wp_hat[:, ip]

        # ldlq and ldlqRG
        return target_fn(
            w.float(), H,
            nbits=config.nbits,
            n_greedy_passes=config.npasses,
            unbiased=config.unbiased,
            blocksize=config.blocksize,
        )

    raise ValueError(f"Unknown ldlq_method: {config.ldlq_method}")


@torch.no_grad()
def quantize_weight_ldlq(
    w: torch.Tensor,
    H: torch.Tensor,
    config: LDLQConfig,
) -> torch.Tensor:
    """
    Main quantization function for QuIP using LDLQ algorithm.

    This function handles:
    1. Pre-processing: Normalize weights based on quant_func (minmax/rms)
    2. LDLQ rounding: Apply Hessian-aware rounding
    3. Post-processing: Denormalize weights

    Args:
        w: Weight tensor [m, d]
        H: Hessian matrix [d, d]
        config: LDLQ configuration

    Returns:
        Quantized weight tensor (float16)
    """
    maxq = 2**config.nbits - 1

    # Special case: minmax + ldl_gptqequiv (GPTQ-compatible path)
    if config.quant_func == "minmax" and config.ldlq_method == "ldl_gptqequiv":
        wr = round_ldl_gptqequiv(
            (w / config.scale) + config.zero, H,
            nbits=config.nbits,
            unbiased=config.unbiased,
        )
        return (config.scale * (wr - config.zero)).half()

    # --- Phase 1: Pre-processing based on quant_func ---
    if config.quant_func == "minmax":
        # MinMax: use pre-computed scale/zero
        wr = torch.clamp((w / config.scale) + config.zero, 0, maxq)
    elif config.quant_func == "rms":
        # RMS: compute scale dynamically
        scale_rms = 2.4 * w.square().mean().sqrt() + 1e-16
        wr = w / scale_rms
        wr = torch.clamp(((wr + 1) / 2) * maxq, 0, maxq)
    else:
        raise ValueError(f"Unknown quant_func: {config.quant_func}")

    # --- Phase 2: LDLQ rounding ---
    wr = round_ldlq(wr, H, config)

    # --- Phase 3: Post-processing based on quant_func ---
    if config.quant_func == "minmax":
        wr = config.scale * (wr - config.zero)
    elif config.quant_func == "rms":
        wr = (wr / maxq) * 2 - 1
        wr = wr * scale_rms

    return wr.half()
