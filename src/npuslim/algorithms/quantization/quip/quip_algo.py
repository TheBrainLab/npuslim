"""Chunk-wise QuIP algorithm for compressor task.

QuIP (Quantization with Incoherence Processing) enhances weight-only
quantization by applying:
1. Diagonal rescaling (weight/Hessian balancing)
2. Random orthogonal projection (butterfly matrix)
3. LDLQ rounding with Hessian-aware error correction

Design mirrors GPTQAlgorithm: strict chunk lifecycle with
calibrate -> quantize -> pack inside each process_chunk call.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import primefac
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from npuslim.algorithms.quantization.hessian import (
    BaseHessianAlgorithm,
    BaseHessianModule,
    _get_child_module,
    _is_transformers_conv1d,
    compute_scales_with_zero,
)
from npuslim.core import AlgorithmRegistry
from npuslim.core.backend import bh


# ---------------------------------------------------------------------------
# Butterfly matrix helpers (from v1 quip_module.py)
# ---------------------------------------------------------------------------

class ButterflyMode(str, Enum):
    BUTTERFLY_PERMUTE = "butterfly_permute"
    BUTTERFLY_PERMUTE_NOBLOCK = "butterfly_permute_noblock"
    BUTTERFLY_NOPERMUTE = "butterfly_nopermute"
    RANDOM_ORTHO = "random_ortho"


def _butterfly_factors(n):
    pf = list(primefac.primefac(n))
    return (math.prod(pf[0::2]), math.prod(pf[1::2]))


def _gen_rand_orthos(m, p):
    if p != 2:
        seed = int(torch.randint(0, 2**31, (1,)).item())
        return torch.tensor(
            scipy.stats.special_ortho_group.rvs(p, size=m, random_state=seed)
        ).to(torch.float32)
    X = torch.zeros(m, 2, 2)
    t = torch.rand(m) * (2 * math.pi)
    sin_t, cos_t = torch.sin(t), torch.cos(t)
    X[:, 0, 0] = cos_t
    X[:, 1, 1] = cos_t
    X[:, 0, 1] = sin_t
    X[:, 1, 0] = -sin_t
    return X


def _gen_rand_ortho_butterfly(n):
    return (
        [_gen_rand_orthos(n // p, p) for p in _butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


def _gen_rand_ortho_butterfly_noblock(n):
    return (
        [_gen_rand_orthos(1, p) for p in _butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


def _gen_rand_ortho_butterfly_nopermute(n):
    return (
        [_gen_rand_orthos(n // p, p) for p in _butterfly_factors(n)],
        torch.arange(n),
        torch.arange(n),
    )


def _mul_ortho_butterfly(Bpp, x):
    (B, p_in, p_out) = Bpp
    orig_dim = 2
    if len(x.shape) == 1:
        (n,) = x.shape
        x = x.reshape(n, 1)
        orig_dim = 1
    (n, q) = x.shape
    x = x[p_in, :]
    pfn = tuple(_butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i + 1):])
        x = x.reshape(mpfx, p, msfx, q).permute(0, 2, 1, 3).reshape(mpfx * msfx, p, q)
        x = B[i] @ x
        x = x.reshape(mpfx, msfx, p, q).permute(0, 2, 1, 3).reshape(n, q)
    x = x[p_out, :]
    if orig_dim == 1:
        x = x.reshape(n)
    return x


def _rand_ortho_butterfly(n):
    return _mul_ortho_butterfly(_gen_rand_ortho_butterfly(n), torch.eye(n))


def _rand_ortho_butterfly_noblock(n):
    return _mul_ortho_butterfly(_gen_rand_ortho_butterfly_noblock(n), torch.eye(n))


def _rand_ortho_butterfly_nopermute(n):
    return _mul_ortho_butterfly(_gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))


def _rand_ortho_matrix(n):
    return torch.tensor(scipy.stats.special_ortho_group.rvs(n), dtype=torch.float32)


def _generate_proj_matrix(mode: ButterflyMode, size: int) -> torch.Tensor:
    """Generate a projection matrix for the given butterfly mode and size."""
    if mode == ButterflyMode.BUTTERFLY_PERMUTE:
        return _rand_ortho_butterfly(size).to(torch.float32)
    elif mode == ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK:
        return _rand_ortho_butterfly_noblock(size).to(torch.float32)
    elif mode == ButterflyMode.BUTTERFLY_NOPERMUTE:
        return _rand_ortho_butterfly_nopermute(size).to(torch.float32)
    elif mode == ButterflyMode.RANDOM_ORTHO:
        return _rand_ortho_matrix(size).to(torch.float32)
    raise ValueError(f"Unknown projection mode: {mode}")


# ---------------------------------------------------------------------------
# LDLQ rounding algorithms (from v1 vector_balance.py)
# ---------------------------------------------------------------------------

@dataclass
class LDLQConfig:
    nbits: int = 4
    quant_func: Literal["minmax", "rms"] = "rms"
    ldlq_method: Literal[
        "ldlq", "ldlqRG", "allbal", "ldlbal_admm", "ldl_gptqequiv"
    ] = "ldlq"
    npasses: int = 0
    unbiased: bool = False
    blocksize: int = 128
    scale: Optional[torch.Tensor] = None
    zero: Optional[torch.Tensor] = None


def _check_nbits(wr, nbits):
    wr_vals, _ = torch.unique(wr, sorted=True, return_counts=True)
    assert len(wr_vals) <= 2**nbits


def _allonce(x, w, unbiased=False):
    if unbiased:
        z = torch.floor(w - x + torch.rand(x.shape).to(x.device))
    else:
        z = torch.round(w - x)
    return w - z


def _round_ldl(w, H, nbits, n_greedy_passes=9, unbiased=False):
    assert (not unbiased) or (n_greedy_passes == 0)
    _d, d_ = H.shape
    assert _d == d_
    m, d = w.shape
    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    eta = torch.rand(w.shape).to(w.device) if unbiased else 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in reversed(range(d)):
        w_hat[:, i] = torch.clamp(
            torch.floor(w[:, i] + (w[:, i:] - w_hat[:, i:]) @ L[i:, i] + eta[:, i]),
            min=0, max=2**nbits - 1,
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
    _check_nbits(wr, nbits)
    return wr


def _round_ldl_block(w, H, nbits, blocksize=128, n_greedy_passes=9, unbiased=False):
    assert (not unbiased) or (n_greedy_passes == 0)
    _d, d_ = H.shape
    assert _d == d_
    m, d = w.shape
    L = torch.linalg.cholesky(H)
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    eta = torch.rand(w.shape).to(w.device) if unbiased else 0.5 * torch.ones(w.shape).to(w.device)
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
                torch.floor(W1[:, i] + (W1 - WHat1) @ L1[i1:i2, i] + W2Hdiff @ L1[i2:, i] + Eta1[:, i]),
                min=0, max=2**nbits - 1,
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
    _check_nbits(wr, nbits)
    return wr


def _round_sorted_ldlqRG(w, H, nbits, n_greedy_passes=9, unbiased=False, pivot=None):
    p = torch.argsort(torch.diag(H))
    Hp = H[p, :][:, p]
    wp = w[:, p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:, p] = _round_ldl(wp, Hp, nbits, n_greedy_passes, unbiased)
    return wr


def _round_sorted_ldlqRG_block(w, H, nbits, blocksize=128, n_greedy_passes=9, unbiased=False, pivot=None):
    p = torch.argsort(torch.diag(H))
    Hp = H[p, :][:, p]
    wp = w[:, p]
    wr = torch.zeros(w.shape).to(w.device)
    wr[:, p] = _round_ldl_block(wp, Hp, nbits, blocksize, n_greedy_passes, unbiased)
    return wr


def _round_allbal(w, H, nbits, npasses, unbiased=False, calc_entropy=False):
    _d, d_ = H.shape
    assert _d == d_
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
    _check_nbits(wr, nbits)
    return wr


def _round_allbal_block(w, H, nbits, npasses, blocksize=128, unbiased=False, calc_entropy=False):
    _d, d_ = H.shape
    assert _d == d_
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
    _check_nbits(wr, nbits)
    return wr


def _round_ldl_gptqequiv(w, H, nbits, unbiased=False):
    _d, d_ = H.shape
    assert _d == d_
    m, d = w.shape
    H = torch.flip(H, [0, 1])
    L = torch.linalg.cholesky(H)
    L = torch.flip(L, [0, 1])
    L = L @ torch.diag(1 / torch.diag(L))
    L = L - torch.eye(d, device=L.device)
    eta = torch.rand(w.shape).to(w.device) if unbiased else 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in range(d):
        w_hat[:, i] = torch.clamp(
            torch.floor(w[:, i] + (w[:, :i+1] - w_hat[:, :i+1]) @ L[:i+1, i] + eta[:, i]),
            min=0, max=2**nbits - 1,
        )
    _check_nbits(w_hat, nbits)
    return w_hat


def _round_ldlq(w: torch.Tensor, H: torch.Tensor, config: LDLQConfig) -> torch.Tensor:
    method_registry = {
        "ldlq": (_round_ldl, _round_ldl_block),
        "ldlqRG": (_round_sorted_ldlqRG, _round_sorted_ldlqRG_block),
        "allbal": (_round_allbal, _round_allbal_block),
    }
    if config.ldlq_method == "ldlbal_admm":
        p = torch.argsort(torch.diag(H))
        Hp = H[p, :][:, p]
        wp = w[:, p]
        wr = torch.zeros(w.shape).to(w.device)
        wr[:, p] = _round_ldl(wp, Hp, config.nbits, config.npasses, config.unbiased)
        return wr
    if config.ldlq_method == "ldl_gptqequiv":
        return _round_ldl_gptqequiv(w, H, config.nbits, config.unbiased)

    if config.ldlq_method in method_registry:
        std_fn, block_fn = method_registry[config.ldlq_method]
        target_fn = block_fn if w.shape[1] > config.blocksize else std_fn
        if config.ldlq_method == "allbal":
            Hdiag = H.diag()
            p = Hdiag.sort(descending=True).indices
            Hp = H[:, p][p, :]
            wp = w[:, p]
            wp_hat = target_fn(
                wp, Hp, nbits=config.nbits, npasses=config.npasses,
                unbiased=config.unbiased, blocksize=config.blocksize,
            )
            ip = torch.argsort(p.float()) if bh.has_npu else torch.argsort(p)
            return wp_hat[:, ip]
        return target_fn(
            w.float(), H, nbits=config.nbits, n_greedy_passes=config.npasses,
            unbiased=config.unbiased, blocksize=config.blocksize,
        )
    raise ValueError(f"Unknown ldlq_method: {config.ldlq_method}")


@torch.no_grad()
def _quantize_weight_ldlq(
    w: torch.Tensor,
    H: torch.Tensor,
    config: LDLQConfig,
    return_int: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    maxq = 2**config.nbits - 1
    scale_rms = None

    if config.quant_func == "minmax" and config.ldlq_method == "ldl_gptqequiv":
        wr = _round_ldl_gptqequiv(
            (w / config.scale) + config.zero, H,
            nbits=config.nbits, unbiased=config.unbiased,
        )
        quant_w = (config.scale * (wr - config.zero)).half()
        return (quant_w, wr.to(torch.int32)) if return_int else quant_w

    if config.quant_func == "minmax":
        wr = torch.clamp((w / config.scale) + config.zero, 0, maxq)
    elif config.quant_func == "rms":
        scale_rms = 2.4 * w.square().mean().sqrt() + 1e-16
        wr = w / scale_rms
        wr = torch.clamp(((wr + 1) / 2) * maxq, 0, maxq)
    else:
        raise ValueError(f"Unknown quant_func: {config.quant_func}")

    wr = _round_ldlq(wr, H, config)
    w_int = wr.to(torch.int32)

    if config.quant_func == "minmax":
        wr = config.scale * (wr - config.zero)
    elif config.quant_func == "rms":
        wr = (wr / maxq) * 2 - 1
        wr = wr * scale_rms

    quant_w = wr.half()
    return (quant_w, w_int) if return_int else quant_w


# ---------------------------------------------------------------------------
# QuIPModule — per-linear-layer QuIP quantization handler
# ---------------------------------------------------------------------------

class QuIPModule(BaseHessianModule):
    def __init__(
        self,
        layer: nn.Module,
        *,
        wbits: int = 4,
        quant_func: str = "rms",
        ldlq_method: str = "ldlq",
        npasses: int = 0,
        unbiased: bool = False,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        preproc_rescale: bool = True,
        preproc_proj: bool = True,
        preproc_proj_mode: int = 2,
    ):
        super().__init__(layer=layer, percdamp=percdamp, preproc_hessian=preproc_hessian)
        self.wbits = int(wbits)
        self.quant_func = quant_func
        self.ldlq_method = ldlq_method
        self.npasses = int(npasses)
        self.unbiased = bool(unbiased)
        self.blocksize = int(blocksize)
        self.maxq = 2**self.wbits - 1
        self.preproc_rescale = bool(preproc_rescale)
        self.preproc_proj = bool(preproc_proj)
        self.preproc_proj_mode = self._resolve_proj_mode(preproc_proj_mode)

        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.scale_rms: Optional[torch.Tensor] = None
        self.scaleWH: Optional[torch.Tensor] = None
        self.projU: Optional[torch.Tensor] = None
        self.projV: Optional[torch.Tensor] = None
        self.proj_seed_u: int = 0
        self.proj_seed_v: int = 0

    @staticmethod
    def _resolve_proj_mode(mode):
        if isinstance(mode, ButterflyMode):
            return mode
        mode_map = {
            0: ButterflyMode.BUTTERFLY_PERMUTE,
            1: ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK,
            2: ButterflyMode.BUTTERFLY_NOPERMUTE,
            3: ButterflyMode.RANDOM_ORTHO,
        }
        return mode_map.get(int(mode), ButterflyMode.BUTTERFLY_NOPERMUTE)

    def preproc(self) -> None:
        if self.preproc_rescale:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.to(torch.float32)
            H /= H.abs().max()
            diagH = torch.clamp(torch.diag(H), min=1e-8)
            diagW2 = torch.clamp(torch.diag(w.T @ w), min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().clamp(min=1e-8).to(torch.float32)
            w = w * scaleWH[None, :]
            H = H / scaleWH[None, :] / scaleWH[:, None]
            self.scaleWH = scaleWH.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        if self.preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            self.proj_seed_u = int(torch.randint(0, 2**31, (1,)).item())
            self.proj_seed_v = int(torch.randint(0, 2**31, (1,)).item())
            torch.manual_seed(self.proj_seed_u)
            np.random.seed(self.proj_seed_u % (2**32))
            U = _generate_proj_matrix(self.preproc_proj_mode, w.shape[0]).to(w.device)
            torch.manual_seed(self.proj_seed_v)
            np.random.seed(self.proj_seed_v % (2**32))
            V = _generate_proj_matrix(self.preproc_proj_mode, w.shape[1]).to(w.device)
            H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(H.shape[0], device=w.device)
            w = U @ w @ V.T
            H = V @ H @ V.T
            self.projU = U.cpu()
            self.projV = V.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        super().preproc()

    def postproc(self) -> None:
        if self.preproc_proj and self.projU is not None and self.projV is not None:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            U = self.projU.to(w.device)
            V = self.projV.to(w.device)
            w = U.T @ w @ V
            H = V.T @ H @ V
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        if self.preproc_rescale and self.scaleWH is not None:
            w = self.layer.weight.data.clone()
            H = self.H.data.clone()
            scaleWH = self.scaleWH.to(w.device)
            w = w / scaleWH[None, :]
            H = H * scaleWH[:, None] * scaleWH[None, :]
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        self.preproc_done = True

    def free(self) -> None:
        super().free()
        self.scaleWH = None
        self.projU = None
        self.projV = None

    def fasterquant(self, **kwargs) -> Dict[str, Any]:
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if _is_transformers_conv1d(self.layer):
            w = w.t()
        full_w = w.clone()

        if self.quant_func == "minmax":
            if self.scale is None:
                self.scale, self.zero = compute_scales_with_zero(w, bits=self.wbits, sym=False)
        else:
            self.scale_rms = 2.4 * w.square().mean().sqrt() + 1e-16

        H = self.H.data.clone()
        ldlq_config = LDLQConfig(
            nbits=self.wbits, quant_func=self.quant_func,
            ldlq_method=self.ldlq_method, npasses=self.npasses,
            unbiased=self.unbiased, blocksize=self.blocksize,
            scale=self.scale, zero=self.zero,
        )
        quant_w, w_int = _quantize_weight_ldlq(w=w, H=H, config=ldlq_config, return_int=True)

        if _is_transformers_conv1d(self.layer):
            quant_w = quant_w.t()
        self.layer.weight.data = quant_w.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        quant_w_preproc = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            quant_w_preproc = quant_w_preproc.flatten(1)
        if _is_transformers_conv1d(self.layer):
            quant_w_preproc = quant_w_preproc.t()
        norm_loss = float(torch.norm(quant_w_preproc - full_w).item())
        hess_err = float(((full_w - quant_w_preproc) @ H.float() @ (full_w - quant_w_preproc).T).trace().item())

        self.postproc()

        result = self._collect_quant_params(w_int)
        self.last_metrics = {
            "rows": self.rows,
            "columns": self.columns,
            "hess_err": hess_err / max(self.nsamples, 1),
            "norm_loss": norm_loss,
        }
        self.free()
        return result

    def _collect_quant_params(self, w_int: torch.Tensor) -> Dict[str, Any]:
        scaleWH = self.scaleWH
        if scaleWH is None:
            scaleWH = torch.ones(self.columns, dtype=torch.float32)
        else:
            scaleWH = scaleWH.float()

        if self.quant_func == "minmax":
            scales = self.scale.half()
            zeros = self.zero.half()
        else:
            scales = self.scale_rms.half().view(1)
            zeros = None

        return {
            "w_int": w_int.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu() if zeros is not None else None,
            "scaleWH": scaleWH.cpu(),
            "proj_seed_u": self.proj_seed_u,
            "proj_seed_v": self.proj_seed_v,
            "proj_mode": self.preproc_proj_mode.value if isinstance(self.preproc_proj_mode, ButterflyMode) else int(self.preproc_proj_mode),
            "bias": self.layer.bias.data.cpu() if self.layer.bias is not None else None,
        }


# ---------------------------------------------------------------------------
# QuIPLinear — packed quantized linear layer for deployment
# ---------------------------------------------------------------------------

class QuIPLinear(nn.Module):
    def __init__(
        self,
        bits: int,
        infeatures: int,
        outfeatures: int,
        has_zero: bool = True,
        bias: bool = False,
        proj_mode: int = 2,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.has_zero = has_zero
        self.maxq = 2**self.bits - 1
        self.proj_mode = proj_mode

        self.register_buffer(
            "qweight", torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        if has_zero:
            self.register_buffer("scales", torch.zeros((outfeatures, 1), dtype=torch.float16))
            self.register_buffer("zeros", torch.zeros((outfeatures, 1), dtype=torch.float16))
        else:
            self.register_buffer("scales", torch.zeros(1, dtype=torch.float16))
            self.zeros = None

        self.register_buffer("scaleWH", torch.zeros(infeatures, dtype=torch.float32))
        self.register_buffer("proj_seed_u", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("proj_seed_v", torch.tensor(0, dtype=torch.int64))

        self._cached_projU = None
        self._cached_projV = None

        if bias:
            self.register_buffer("bias", torch.zeros(outfeatures, dtype=torch.float16))
        else:
            self.bias = None

        if self.bits in [2, 4, 8]:
            self.register_buffer(
                "wf",
                torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0),
                persistent=False,
            )
        elif self.bits == 3:
            self.register_buffer(
                "wf",
                torch.tensor(
                    [[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                     [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                     [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0]],
                    dtype=torch.int32,
                ).reshape(1, 3, 12),
                persistent=False,
            )

    def _generate_butterfly_from_seed(self, seed: int, size: int) -> torch.Tensor:
        mode_map = {
            "butterfly_permute": ButterflyMode.BUTTERFLY_PERMUTE,
            "butterfly_permute_noblock": ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK,
            "butterfly_nopermute": ButterflyMode.BUTTERFLY_NOPERMUTE,
            "random_ortho": ButterflyMode.RANDOM_ORTHO,
        }
        mode_map_int = {
            0: ButterflyMode.BUTTERFLY_PERMUTE,
            1: ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK,
            2: ButterflyMode.BUTTERFLY_NOPERMUTE,
            3: ButterflyMode.RANDOM_ORTHO,
        }
        if isinstance(self.proj_mode, str):
            mode = mode_map.get(self.proj_mode, ButterflyMode.BUTTERFLY_NOPERMUTE)
        else:
            mode = mode_map_int.get(int(self.proj_mode), ButterflyMode.BUTTERFLY_NOPERMUTE)

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        return _generate_proj_matrix(mode, size).to(torch.float32)

    def _unpack_weights(self) -> torch.Tensor:
        if self.bits in [2, 4, 8]:
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int8)
            weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        elif self.bits == 3:
            weight = self.qweight.reshape(
                self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
            ).expand(-1, -1, 12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1,
            )
        else:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")
        return weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    def _dequantize(self, weight_int: torch.Tensor) -> torch.Tensor:
        if self.has_zero and self.zeros is not None:
            w = self.scales.T * (weight_int.float() - self.zeros.T)
        else:
            w = ((weight_int.float() / self.maxq) * 2 - 1) * self.scales
            w = w.T
        return w.to(torch.float32)

    def _postproc(self, w: torch.Tensor) -> torch.Tensor:
        if self._cached_projU is None:
            self._cached_projU = self._generate_butterfly_from_seed(
                int(self.proj_seed_u.item()), self.outfeatures,
            ).to(w.device)
        if self._cached_projV is None:
            self._cached_projV = self._generate_butterfly_from_seed(
                int(self.proj_seed_v.item()), self.infeatures,
            ).to(w.device)
        U = self._cached_projU
        V = self._cached_projV
        scaleWH = self.scaleWH.to(w.device)
        w = U.T @ w @ V
        w = w / scaleWH.unsqueeze(0).clamp(min=1e-8)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        if self.wf.device != self.qweight.device:
            self.wf = self.wf.to(self.qweight.device)
        weight_int = self._unpack_weights()
        w = self._dequantize(weight_int)
        w = self._postproc(w)
        bias = self.bias.to(x_dtype) if self.bias is not None else None
        out = F.linear(x, w.to(x_dtype), bias)
        return out.reshape(out_shape)

    def pack(
        self,
        w_int: torch.Tensor,
        scales: torch.Tensor,
        zeros: Optional[torch.Tensor],
        scaleWH: torch.Tensor,
        proj_seed_u: int,
        proj_seed_v: int,
        bias: Optional[torch.Tensor] = None,
    ):
        self.scales.copy_(scales)
        if zeros is not None and self.zeros is not None:
            self.zeros.copy_(zeros)
        self.scaleWH.copy_(scaleWH)
        self.proj_seed_u.fill_(proj_seed_u)
        self.proj_seed_v.fill_(proj_seed_v)
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)

        w_int_t = w_int.t().contiguous()
        w_int_np = w_int_t.numpy().astype(np.uint32)
        qweight = np.zeros(
            (w_int_np.shape[0] // 32 * self.bits, w_int_np.shape[1]), dtype=np.uint32,
        )
        i, row = 0, 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= w_int_np[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i))
                i += 10
                qweight[row] |= w_int_np[i] << 30
                row += 1
                qweight[row] |= (w_int_np[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= w_int_np[i] << 31
                row += 1
                qweight[row] |= (w_int_np[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= w_int_np[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")
        self.qweight.copy_(torch.from_numpy(qweight.astype(np.int32)))

    def clear_cache(self):
        self._cached_projU = None
        self._cached_projV = None


# ---------------------------------------------------------------------------
# QuIPAlgorithm — streaming chunk-based algorithm
# ---------------------------------------------------------------------------

@AlgorithmRegistry.register("QuIP", aliases=["quip"])
class QuIPAlgorithm(BaseHessianAlgorithm):
    """Chunk-wise QuIP algorithm for compressor task."""

    _TAG = "QuIP"
    _quantized_type_label = "QuIP"

    def __init__(
        self,
        wbits: int = 4,
        w_bits: Optional[int] = None,
        groupsize: int = -1,
        group_size: Optional[int] = None,
        quant_func: str = "rms",
        ldlq_method: str = "ldlq",
        npasses: int = 0,
        unbiased: bool = False,
        blocksize: int = 128,
        percdamp: float = 0.01,
        preproc_hessian: bool = True,
        preproc_rescale: bool = True,
        preproc_proj: bool = True,
        preproc_proj_mode: int = 2,
        incoh_processing: bool = True,
        fake_quant: bool = False,
        max_calib_samples: int = 128,
        **kwargs,
    ):
        if w_bits is not None:
            wbits = int(w_bits)
        if group_size is not None:
            groupsize = int(group_size)
        super().__init__(max_calib_samples=max_calib_samples, **kwargs)
        self.wbits = int(wbits)
        self.groupsize = int(groupsize)
        self.quant_func = quant_func
        self.ldlq_method = ldlq_method
        self.npasses = int(npasses)
        self.unbiased = bool(unbiased)
        self.blocksize = int(blocksize)
        self.percdamp = float(percdamp)
        self.preproc_hessian = bool(preproc_hessian)
        self.preproc_rescale = bool(preproc_rescale)
        self.preproc_proj = bool(preproc_proj)
        self.preproc_proj_mode = int(preproc_proj_mode)
        self.fake_quant = bool(fake_quant)

        if incoh_processing:
            self.quant_func = "rms"
            self.preproc_hessian = True
            self.preproc_rescale = True
            self.preproc_proj = True

    @property
    def _ascend_quant_type(self) -> str:
        return "FLOAT" if self.fake_quant else f"W{self.wbits}A16"

    def _log_start_params(self) -> None:
        logger.info(
            f"[{self._TAG}] start: wbits={self.wbits}, quant_func={self.quant_func}, "
            f"ldlq_method={self.ldlq_method}, incoh=(rescale={self.preproc_rescale}, proj={self.preproc_proj})"
        )

    def _pack_quip_linear_tensors(
        self,
        module_name: str,
        linear_module: nn.Module,
        result: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        has_zero = result["zeros"] is not None
        proj_mode = result["proj_mode"]
        if isinstance(proj_mode, ButterflyMode):
            proj_mode = proj_mode.value

        quip_linear = QuIPLinear(
            bits=self.wbits,
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            has_zero=has_zero,
            bias=result["bias"] is not None,
            proj_mode=proj_mode,
        )
        quip_linear.cpu()
        quip_linear.pack(
            w_int=result["w_int"],
            scales=result["scales"],
            zeros=result["zeros"],
            scaleWH=result["scaleWH"],
            proj_seed_u=result["proj_seed_u"],
            proj_seed_v=result["proj_seed_v"],
            bias=result["bias"],
        )

        tensors: Dict[str, torch.Tensor] = {}
        quantized_names: List[str] = []
        tensors[f"{module_name}.qweight"] = quip_linear.qweight.cpu()
        tensors[f"{module_name}.scales"] = quip_linear.scales.cpu()
        quantized_names.extend([f"{module_name}.qweight", f"{module_name}.scales"])

        if has_zero and quip_linear.zeros is not None:
            tensors[f"{module_name}.zeros"] = quip_linear.zeros.cpu()
            quantized_names.append(f"{module_name}.zeros")

        tensors[f"{module_name}.scaleWH"] = quip_linear.scaleWH.cpu()
        tensors[f"{module_name}.proj_seed_u"] = quip_linear.proj_seed_u.cpu()
        tensors[f"{module_name}.proj_seed_v"] = quip_linear.proj_seed_v.cpu()
        quantized_names.extend([
            f"{module_name}.scaleWH",
            f"{module_name}.proj_seed_u",
            f"{module_name}.proj_seed_v",
        ])

        if quip_linear.bias is not None:
            tensors[f"{module_name}.bias"] = quip_linear.bias.cpu()
        return tensors, quantized_names

    def _create_handlers(self, layer_module: nn.Module, targets) -> Dict[str, QuIPModule]:
        handlers: Dict[str, QuIPModule] = {}
        for module_rel_name, *_ in targets:
            submodule = _get_child_module(layer_module, module_rel_name)
            if not isinstance(submodule, nn.Linear):
                continue
            handlers[module_rel_name] = QuIPModule(
                submodule,
                wbits=self.wbits,
                quant_func=self.quant_func,
                ldlq_method=self.ldlq_method,
                npasses=self.npasses,
                unbiased=self.unbiased,
                blocksize=self.blocksize,
                percdamp=self.percdamp,
                preproc_hessian=self.preproc_hessian,
                preproc_rescale=self.preproc_rescale,
                preproc_proj=self.preproc_proj,
                preproc_proj_mode=self.preproc_proj_mode,
            )
        return handlers

    def _process_layer_handlers(self, layer, targets, handlers, chunk) -> tuple[set[str], int]:
        quantized_tensor_names: set[str] = set()
        quantized_weights = 0
        quant_results = []
        for module_rel_name, rel_weight_name, rel_bias_name, _weight_tensor, _bias_tensor, *rest in targets:
            is_3d = rest[0] if rest else False
            handler = handlers.get(module_rel_name)
            if handler is None:
                continue
            result = handler.fasterquant(layer_name=f"{layer.name}.{module_rel_name}")
            metrics = getattr(handler, "last_metrics", {})
            if metrics:
                full_name = f"{layer.name}.{module_rel_name}"
                logger.info(
                    f"[{self._TAG}] {full_name:<50s} | "
                    f"shape=[{int(metrics.get('rows', 0)):>5},{int(metrics.get('columns', 0)):>5}] | "
                    f"hess_err={float(metrics.get('hess_err', 0.0)):<12.6f} | "
                    f"norm_loss={float(metrics.get('norm_loss', 0.0)):<12.6f}"
                )
            quant_results.append((module_rel_name, rel_weight_name, rel_bias_name, result, handler))

        pack_iter = tqdm(
            quant_results,
            total=len(quant_results),
            desc=f"{self._TAG.lower()} pack c{chunk.chunk_index} {layer.name}",
            leave=True,
            disable=len(quant_results) <= 1,
        )
        for module_rel_name, rel_weight_name, rel_bias_name, result, handler in pack_iter:
            if self.fake_quant:
                layer.tensors[rel_weight_name] = (
                    handler.layer.weight.detach().to(layer.tensors[rel_weight_name].dtype).cpu()
                )
                quantized_tensor_names.add(f"{layer.name}.{rel_weight_name}")
            else:
                packed_tensors, packed_quant_names = self._pack_quip_linear_tensors(
                    module_name=module_rel_name,
                    linear_module=handler.layer,
                    result=result,
                )
                layer.tensors.pop(rel_weight_name, None)
                layer.tensors.pop(rel_bias_name, None)
                for rel_name, tensor in packed_tensors.items():
                    layer.tensors[rel_name] = tensor
                for rel_quant_name in packed_quant_names:
                    quantized_tensor_names.add(f"{layer.name}.{rel_quant_name}")
            quantized_weights += 1
        return quantized_tensor_names, quantized_weights

    def _update_quantization_metadata(self) -> None:
        if self._model_config is None:
            return
        if self.target_backend == "npu":
            self._model_config.ascend_quant_config = {
                "model_quant_type": self._ascend_quant_type,
                "group_size": self.groupsize,
                "quant_layer_types": ["QuIPLinear"],
                "include_g_idx": False,
                "has_offset": True,
            }
            if hasattr(self._model_config, "quantization_config"):
                try:
                    delattr(self._model_config, "quantization_config")
                except Exception:
                    pass
        else:
            self._model_config.quantization_config = {
                "bits": self.wbits,
                "quant_func": self.quant_func,
                "quant_method": "quip",
                "checkpoint_format": "quip",
                "preproc_proj_mode": self.preproc_proj_mode,
            }
        self._mark_model_quantized()
        logger.info(f"[{self._TAG}] model quantization metadata updated")
