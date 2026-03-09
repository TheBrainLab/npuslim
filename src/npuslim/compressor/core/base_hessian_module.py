import math
import torch
import torch.nn as nn
import transformers
from loguru import logger

from npuslim.utils.backend import bh


DEBUG = False


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
        """
        Preprocessing for Hessian-based quantization.

        Handles shared preprocessing logic (preproc_hessian).
        Subclasses can override to add algorithm-specific preprocessing.
        """
        percdamp = self.percdamp
        preproc_hessian = self.preproc_hessian

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
        Post-processing to reverse preprocessing transformations.

        Base implementation is a no-op.
        Subclasses should override to reverse algorithm-specific preprocessing.
        """
        assert self.preproc_done is True

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        bh.empty_cache()

    def print_log(self, avg_loss, norm_loss):
        w_shape = self.layer.weight.shape
        w_out, w_in = w_shape[0], w_shape[1]
        label_width = 25
        logger.info(f"{'Layer Shape:':<{label_width}} [Out={w_out}, In={w_in}]")
        logger.info(f"{'Avg Loss (Hessian):':<{label_width}} {avg_loss:.6f}")
        logger.info(f"{'Norm Loss (L2):':<{label_width}} {norm_loss:.6f}")
