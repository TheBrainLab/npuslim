"""
QuIP (Quantization with Incoherence Processing) Module.

This implementation follows the original QuIP paper using:
1. Balance/LDLQ algorithm for quantization
2. Incoherence preprocessing (rescale + projection)
3. Supports both fake quantization and real quantization

Reference: https://github.com/Cornell-RelaxML/QuIP
"""

import math
import time
from enum import Enum

import numpy as np
import primefac
import scipy
import torch
import torch.nn as nn
import transformers

from npuslim.compressor.core.base_hessian_module import BaseHessianModule
from npuslim.compressor.core.quant_func import compute_scales_with_zero
from .vector_balance import quantize_weight_ldlq, LDLQConfig


class ButterflyMode(str, Enum):
    """Butterfly matrix generation modes for orthogonal projection"""

    # 2-factor butterfly + permutation + blocking
    BUTTERFLY_PERMUTE = "butterfly_permute"
    # 2-factor butterfly + permutation (default, faster)
    BUTTERFLY_PERMUTE_NOBLOCK = "butterfly_permute_noblock"
    # 2-factor butterfly only (no permutation)
    BUTTERFLY_NOPERMUTE = "butterfly_nopermute"
    # Random orthogonal matrix (slower but more general)
    RANDOM_ORTHO = "random_ortho"


def butterfly_factors(n):
    pf = list(primefac.primefac(n))
    return (math.prod(pf[0::2]), math.prod(pf[1::2]))


def gen_rand_orthos(m, p):
    if p != 2:
        # Use torch's RNG to generate seed for scipy (deterministic with torch.manual_seed)
        seed = int(torch.randint(0, 2**31, (1,)).item())
        return torch.tensor(scipy.stats.special_ortho_group.rvs(p, size=m, random_state=seed)).to(
            torch.float32
        )
    X = torch.zeros(m, 2, 2)
    t = torch.rand(m) * (2 * math.pi)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    X[:, 0, 0] = cos_t
    X[:, 1, 1] = cos_t
    X[:, 0, 1] = sin_t
    X[:, 1, 0] = -sin_t
    return X


def gen_rand_ortho_butterfly(n):
    """Generate a random orthogonal butterfly matrix of dimension n."""
    return (
        [gen_rand_orthos(n // p, p) for p in butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


def gen_rand_ortho_butterfly_noblock(n):
    """Generate a random orthogonal butterfly matrix of dimension n, without blocking."""
    return (
        [gen_rand_orthos(1, p) for p in butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


def gen_rand_ortho_butterfly_nopermute(n):
    """Generate a random orthogonal butterfly matrix of dimension n, no permutation, but yes blocking."""
    return (
        [gen_rand_orthos(n // p, p) for p in butterfly_factors(n)],
        torch.arange(n),
        torch.arange(n),
    )


def mul_ortho_butterfly(Bpp, x):
    """Multiply by a random orthogonal butterfly matrix."""
    (B, p_in, p_out) = Bpp
    assert (len(x.shape) == 1) or (len(x.shape) == 2)
    orig_dim = 2
    if len(x.shape) == 1:
        (n,) = x.shape
        x = x.reshape(n, 1)
        orig_dim = 1
    (n, q) = x.shape
    x = x[p_in, :]
    pfn = tuple(butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i + 1) :])
        x = x.reshape(mpfx, p, msfx, q).permute(0, 2, 1, 3).reshape(mpfx * msfx, p, q)
        x = B[i] @ x
        x = x.reshape(mpfx, msfx, p, q).permute(0, 2, 1, 3).reshape(n, q)
    x = x[p_out, :]
    if orig_dim == 1:
        x = x.reshape(n)
    return x


def rand_ortho_butterfly(n):
    """Generate a random orthogonal butterfly matrix and convert it to a dense matrix."""
    return mul_ortho_butterfly(gen_rand_ortho_butterfly(n), torch.eye(n))


def rand_ortho_butterfly_noblock(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(n), torch.eye(n))


def rand_ortho_butterfly_nopermute(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))


def rand_ortho_matrix(n):
    """Generate a random orthogonal matrix of dimension n using scipy."""
    return torch.tensor(scipy.stats.special_ortho_group.rvs(n), dtype=torch.float32)


class QuIPModule(BaseHessianModule):
    """
    QuIP Module using LDLQ/Vector Balance algorithm.

    Following original QuIP's Balance class (bal.py):
    - Uses quantize_weight_ldlq for quantization
    - Supports quant_func="minmax" or "rms"
    - Returns quantization parameters for real quantization
    """

    def __init__(self, *args, **kwargs):
        super(QuIPModule, self).__init__(*args, **kwargs)

        # QuIP parameters
        self.w_bits = getattr(self.config, "w_bits", 4)
        self.npasses = getattr(self.config, "npasses", 0)  # greedy passes
        self.unbiased = getattr(self.config, "unbiased", False)
        self.quant_func = getattr(self.config, "quant_func", "rms")  # "minmax" or "rms"
        self.ldlq_method = getattr(self.config, "ldlq_method", "ldlq")
        self.blocksize = getattr(self.config, "blocksize", 128)
        self.fake_quant = getattr(self.config, "fake_quant", True)

        # Scale/zero for minmax mode (computed in fasterquant)
        self.scale = None
        self.zero = None
        self.maxq = 2**self.w_bits - 1

        # RMS scale (computed in fasterquant for rms mode)
        self.scale_rms = None

        # QuIP-specific preprocessing state
        self.scaleWH = None
        self.projU = None
        self.projV = None
        self.proj_seed_u = 0
        self.proj_seed_v = 0

    def _apply_config(self):
        """Apply QuIP-specific configuration options."""
        super()._apply_config()
        self.preproc_rescale = getattr(self.config, "preproc_rescale", False)
        self.preproc_proj = getattr(self.config, "preproc_proj", False)

        # Convert integer mode to ButterflyMode enum for comparison
        # Official QuIP mapping: 0=butterfly_permute, 1=butterfly_noblock, 2=butterfly_nopermute, 3=random_ortho
        preproc_proj_mode = getattr(self.config, "preproc_proj_mode", 1)
        if isinstance(preproc_proj_mode, int):
            mode_map = {
                0: ButterflyMode.BUTTERFLY_PERMUTE,
                1: ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK,
                2: ButterflyMode.BUTTERFLY_NOPERMUTE,
                3: ButterflyMode.RANDOM_ORTHO,
            }
            self.preproc_proj_mode = mode_map.get(preproc_proj_mode, ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK)
        else:
            self.preproc_proj_mode = preproc_proj_mode

    def preproc(self):
        """
        QuIP-specific preprocessing: rescale + projection + hessian damping.

        Order matters: rescale -> proj -> hessian (same as original QuIP)
        """
        percdamp = self.percdamp
        preproc_hessian = self.preproc_hessian
        preproc_rescale = self.preproc_rescale
        preproc_proj = self.preproc_proj
        preproc_proj_mode = self.preproc_proj_mode

        # Step 1: Rescale preprocessing (weight/Hessian balancing)
        if preproc_rescale:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.to(torch.float32)
            H /= H.abs().max()
            diagH = torch.diag(H)
            diagW2 = torch.diag(w.T @ w)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            w *= scaleWH[None, :]
            H /= scaleWH[None, :]
            H /= scaleWH[:, None]
            w = w.to(torch.float32)
            scaleWH = scaleWH.to(torch.float32)
            self.scaleWH = scaleWH.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        # Step 2: Projection preprocessing (incoherence via butterfly matrices)
        if preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)

            # Generate and save seeds for reproducible Butterfly matrix generation
            self.proj_seed_u = int(torch.randint(0, 2**31, (1,)).item())
            self.proj_seed_v = int(torch.randint(0, 2**31, (1,)).item())

            # Generate U matrix with seed (must set both torch and numpy/scipy seeds)
            torch.manual_seed(self.proj_seed_u)
            np.random.seed(self.proj_seed_u % (2**32))
            if preproc_proj_mode == ButterflyMode.BUTTERFLY_PERMUTE:
                U = rand_ortho_butterfly(w.shape[0]).to(torch.float32).to(w.device)
            elif preproc_proj_mode == ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK:
                U = (
                    rand_ortho_butterfly_noblock(w.shape[0])
                    .to(torch.float32)
                    .to(w.device)
                )
            elif preproc_proj_mode == ButterflyMode.BUTTERFLY_NOPERMUTE:
                U = (
                    rand_ortho_butterfly_nopermute(w.shape[0])
                    .to(torch.float32)
                    .to(w.device)
                )
            elif preproc_proj_mode == ButterflyMode.RANDOM_ORTHO:
                U = rand_ortho_matrix(w.shape[0]).to(w.device)
            else:
                raise NotImplementedError(
                    f"Projection mode '{preproc_proj_mode}' is not implemented yet"
                )

            # Generate V matrix with seed (must set both torch and numpy/scipy seeds)
            torch.manual_seed(self.proj_seed_v)
            np.random.seed(self.proj_seed_v % (2**32))
            if preproc_proj_mode == ButterflyMode.BUTTERFLY_PERMUTE:
                V = rand_ortho_butterfly(w.shape[1]).to(torch.float32).to(w.device)
            elif preproc_proj_mode == ButterflyMode.BUTTERFLY_PERMUTE_NOBLOCK:
                V = (
                    rand_ortho_butterfly_noblock(w.shape[1])
                    .to(torch.float32)
                    .to(w.device)
                )
            elif preproc_proj_mode == ButterflyMode.BUTTERFLY_NOPERMUTE:
                V = (
                    rand_ortho_butterfly_nopermute(w.shape[1])
                    .to(torch.float32)
                    .to(w.device)
                )
            elif preproc_proj_mode == ButterflyMode.RANDOM_ORTHO:
                V = rand_ortho_matrix(w.shape[1]).to(w.device)
            else:
                raise NotImplementedError(
                    f"Projection mode '{preproc_proj_mode}' is not implemented yet"
                )

            H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(
                H.shape[0], device=w.device
            )
            H = H.to(torch.float32)
            w = U @ w @ V.T
            H = V @ H @ V.T
            self.projU = U.cpu()
            self.projV = V.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        # Step 3: Hessian preprocessing (damping and dead feature handling)
        if preproc_hessian:
            w = self.layer.weight.data.clone()
            H = self.H.data.clone()
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            w[:, dead] = 0
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        self.preproc_done = True

    def postproc(self):
        """
        Reverse QuIP-specific preprocessing: projection -> rescale.

        Order is reversed from preproc.
        """
        assert self.preproc_done is True

        # Reverse projection first (opposite order from preproc)
        if self.preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            U = self.projU.to(w.device)
            V = self.projV.to(w.device)
            w = U.T @ w @ V
            H = V.T @ H @ V
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

        # Reverse rescale
        if self.preproc_rescale:
            w = self.layer.weight.data.clone()
            H = self.H.data.clone()
            scaleWH = self.scaleWH.to(w.device)
            w = w / scaleWH[None, :]
            H = H * scaleWH[:, None]
            H = H * scaleWH[None, :]
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

    def free(self):
        """Free QuIP-specific resources."""
        super().free()
        self.scaleWH = None
        self.projU = None
        self.projV = None

    def fasterquant(self, **kwargs):
        """
        Execute QuIP quantization using LDLQ/Vector Balance algorithm.

        Following original QuIP's Balance.fasterquant():
        1. Get weight and Hessian
        2. Apply quantize_weight_ldlq
        3. Update layer weight
        4. Apply postproc (reverse preprocessing) - only for fake_quant

        Returns:
            For fake_quant: dummy scale/zero/g_idx
            For real_quant: dict with w_int, scales, zeros, scaleWH, proj_seeds
        """
        # Get weight
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()

        full_w = w.clone()
        tick = time.time()

        # Compute scale/zero for minmax mode, scale_rms for rms mode
        if self.quant_func == "minmax":
            if self.scale is None:
                self.scale, self.zero = compute_scales_with_zero(
                    w, bits=self.w_bits, sym=False
                )
        else:  # rms
            # Compute RMS scale (same as in quantize_weight_ldlq)
            self.scale_rms = 2.4 * w.square().mean().sqrt() + 1e-16

        # Get Hessian
        H = self.H.data.clone()

        # Build LDLQConfig
        ldlq_config = LDLQConfig(
            nbits=self.w_bits,
            quant_func=self.quant_func,
            ldlq_method=self.ldlq_method,
            npasses=self.npasses,
            unbiased=self.unbiased,
            blocksize=self.blocksize,
            scale=self.scale,
            zero=self.zero,
        )

        # Apply LDLQ quantization (returns dequantized float weights and rounded integers)
        quant_w, w_int = quantize_weight_ldlq(w=w, H=H, config=ldlq_config, return_int=True)

        # Update layer weight
        if isinstance(self.layer, transformers.Conv1D):
            quant_w = quant_w.t()

        self.layer.weight.data = quant_w.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Compute error metrics BEFORE postproc (in preprocessed space)
        quant_w_preproc = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            quant_w_preproc = quant_w_preproc.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            quant_w_preproc = quant_w_preproc.t()

        norm_loss = torch.norm(quant_w_preproc - full_w).item()
        self.error = (
            (
                (full_w - quant_w_preproc)
                @ H.type(torch.float)
                @ (full_w - quant_w_preproc).T
            )
            .trace()
            .item()
        )
        self.Hmag = self.H.max().item()

        # Post-processing: reverse incoherence processing
        # For fake_quant: weights stay as float16 in the layer
        # For real_quant: we still need to apply postproc so subsequent layers get correct activations
        # (The real quant params w_int are already saved before postproc)
        self.postproc()

        # Timing
        self.time = time.time() - tick

        self.print_log(avg_loss=self.error / self.nsamples, norm_loss=norm_loss)

        # Collect preprocessing parameters BEFORE free()
        result = self._collect_quant_params(w_int)

        self.free()

        return result

    def _extract_int_weights(self, w: torch.Tensor, quant_w: torch.Tensor) -> torch.Tensor:
        """
        Extract integer weights from the quantization result.

        Args:
            w: Original weights [out, in] (in preprocessed space)
            quant_w: Dequantized weights [out, in]

        Returns:
            Integer weights [out, in] in range [0, maxq]
        """
        if self.quant_func == "minmax":
            # minmax: w_int = round(w / scale) + zero
            if self.scale is None or self.zero is None:
                raise ValueError("scale and zero must be computed for minmax mode")
            w_int = torch.clamp(
                torch.round(w / self.scale) + self.zero,
                0, self.maxq
            )
        else:  # rms
            # rms: w_normalized = w / scale_rms
            #      w_int = round((w_normalized + 1) / 2 * maxq)
            if self.scale_rms is None:
                raise ValueError("scale_rms must be computed for rms mode")
            w_normalized = w / self.scale_rms
            w_int = torch.clamp(
                torch.round(((w_normalized + 1) / 2) * self.maxq),
                0, self.maxq
            )

        return w_int.to(torch.int32)

    def _collect_quant_params(self, w_int: torch.Tensor) -> dict:
        """
        Collect all parameters needed for real quantization.

        Args:
            w_int: Integer weights [out, in]

        Returns:
            dict with all parameters for QuIPLinear.pack()
        """
        # Get preprocessing parameters from BaseHessianModule
        scaleWH = getattr(self, "scaleWH", None)
        proj_seed_u = getattr(self, "proj_seed_u", 0)
        proj_seed_v = getattr(self, "proj_seed_v", 0)
        proj_mode = getattr(self.config, "preproc_proj_mode", 2)

        # Prepare scales and zeros based on quant_func
        if self.quant_func == "minmax":
            if self.scale is None or self.zero is None:
                raise ValueError("scale and zero must be computed for minmax mode")
            scales = self.scale.half()  # [out, 1]
            zeros = self.zero.half()    # [out, 1]
        else:  # rms
            if self.scale_rms is None:
                raise ValueError("scale_rms must be computed for rms mode")
            scales = self.scale_rms.half().view(1)  # scalar
            zeros = None

        # Handle scaleWH
        if scaleWH is None:
            scaleWH = torch.ones(self.columns, dtype=torch.float32)
        else:
            scaleWH = scaleWH.float()

        return {
            "w_int": w_int.cpu(),
            "scales": scales.cpu(),
            "zeros": zeros.cpu() if zeros is not None else None,
            "scaleWH": scaleWH.cpu(),
            "proj_seed_u": proj_seed_u,
            "proj_seed_v": proj_seed_v,
            "proj_mode": proj_mode,
            "bias": self.layer.bias.data.cpu() if self.layer.bias is not None else None,
        }

    def print_log(self, avg_loss, norm_loss):
        """Print quantization statistics."""
        w_shape = self.layer.weight.shape
        w_out, w_in = w_shape[0], w_shape[1]
        label_width = 25
        logger_info = [
            f"{'Layer Shape:':<{label_width}} [Out={w_out}, In={w_in}]",
            f"{'Hessian Error:':<{label_width}} {avg_loss:.6f}",
            f"{'Norm Loss (L2):':<{label_width}} {norm_loss:.6f}",
            f"{'Hessian Max:':<{label_width}} {self.Hmag:.6f}",
            f"{'Time:':<{label_width}} {self.time:.2f}s",
        ]
        from loguru import logger

        for line in logger_info:
            logger.info(line)
