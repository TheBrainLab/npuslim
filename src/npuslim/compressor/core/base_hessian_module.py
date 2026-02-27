import math
import scipy
import torch
import torch.nn as nn
import primefac
import transformers
import numpy as np
from enum import Enum
from loguru import logger

from npuslim.utils.backend import bh


DEBUG = False


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


# generates a random orthogonal butterfly matrix of dimension n
def gen_rand_ortho_butterfly(n):
    return (
        [gen_rand_orthos(n // p, p) for p in butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


# generates a random orthogonal butterfly matrix of dimension n, without blocking
def gen_rand_ortho_butterfly_noblock(n):
    return (
        [gen_rand_orthos(1, p) for p in butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


# generates a random orthogonal butterfly matrix of dimension n, no permutation, but yes blocking
def gen_rand_ortho_butterfly_nopermute(n):
    return (
        [gen_rand_orthos(n // p, p) for p in butterfly_factors(n)],
        torch.arange(n),
        torch.arange(n),
    )


# multiply by a random orthogonal butterfly matrix
def mul_ortho_butterfly(Bpp, x):
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


# generates a random orthogonal butterfly matrix of dimension n
# and converts it to a dense matrix
def rand_ortho_butterfly(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly(n), torch.eye(n))


def rand_ortho_butterfly_noblock(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(n), torch.eye(n))


def rand_ortho_butterfly_nopermute(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))


def rand_ortho_matrix(n):
    """Generate a random orthogonal matrix of dimension n using scipy"""
    return torch.tensor(scipy.stats.special_ortho_group.rvs(n), dtype=torch.float32)


class BaseHessianModule:
    """
    Base class for quantization methods
    """

    def __init__(self, layer, config=None):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.preproc_done = False

        self.config = config
        self._apply_config()

    def _apply_config(self):
        self.percdamp = getattr(self.config, "percdamp", 0.01)
        self.preproc_hessian = getattr(self.config, "preproc_hessian", False)
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

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        inp = inp.to(self.dev, dtype=torch.float32)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def compute_hinv(self, H):
        percdamp = self.percdamp
        step = 0.01
        Hinv = None

        # 判断是否已经在 preproc 阶段加过阻尼
        # 如果加过，初始的额外阻尼设为 0；否则设为 percdamp
        is_damped_in_preproc = self.preproc_hessian

        # current_percdamp 是本次循环要 *额外* 加上去的阻尼比例
        if is_damped_in_preproc:
            current_percdamp = 0.0  # 第一次尝试完全不加新阻尼
        else:
            current_percdamp = percdamp  # 第一次尝试加上基础阻尼

        while current_percdamp < 1.0:
            try:
                H_try = H.clone()
                if current_percdamp > 0:
                    damp = current_percdamp * torch.mean(torch.diag(H_try))
                    if damp == 0:
                        damp = 1e-5

                    diag_idx = torch.arange(self.columns, device=self.dev)
                    H_try[diag_idx, diag_idx] += damp

                # --- 核心计算 (NPU/GPU 分支) ---
                if bh.name == "npu":
                    try:
                        H_cpu = H_try.to("cpu")
                        L_cpu = torch.linalg.cholesky(H_cpu)
                        inv_L_cpu = torch.cholesky_inverse(L_cpu)
                        Hinv_cpu = torch.linalg.cholesky(inv_L_cpu, upper=True)
                        Hinv = Hinv_cpu.to(self.dev)
                        break
                    except RuntimeError:
                        raise torch._C._LinAlgError("CPU Cholesky failed")
                else:
                    L = torch.linalg.cholesky(H_try)
                    inv_L = torch.cholesky_inverse(L)
                    Hinv = torch.linalg.cholesky(inv_L, upper=True)
                    break

            except (RuntimeError, torch._C._LinAlgError):
                # 失败了，增加阻尼重试
                # 每次增加 step
                if current_percdamp == 0:
                    # 如果第一次是因为没加阻尼挂了，立刻加上基础阻尼
                    current_percdamp = percdamp
                else:
                    current_percdamp += step
                continue

        if Hinv is None:
            raise RuntimeError("Hessian inversion failed even with max damping.")

        return Hinv

    def preproc(self):
        percdamp = self.percdamp
        preproc_hessian = self.preproc_hessian
        preproc_rescale = self.preproc_rescale
        preproc_proj = self.preproc_proj
        preproc_proj_mode = self.preproc_proj_mode

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
        assert self.preproc_done is True
        if self.preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            U = self.projU.to(w.device)
            V = self.projV.to(w.device)
            w = U.T @ w @ V
            H = V.T @ H @ V
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

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
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        self.scaleWH = None
        self.projU = None
        self.projV = None
        bh.empty_cache()

    def print_log(self, avg_loss, norm_loss):
        w_shape = self.layer.weight.shape
        w_out, w_in = w_shape[0], w_shape[1]
        label_width = 25
        logger.info(f"{'Layer Shape:':<{label_width}} [Out={w_out}, In={w_in}]")
        logger.info(f"{'Avg Loss (Hessian):':<{label_width}} {avg_loss:.6f}")
        logger.info(f"{'Norm Loss (L2):':<{label_width}} {norm_loss:.6f}")
