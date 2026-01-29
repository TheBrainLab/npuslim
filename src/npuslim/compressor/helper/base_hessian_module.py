from abc import ABC, abstractmethod
from loguru import logger
import torch
import math
import transformers
from npuslim.utils.backend import bh


class BaseHessianModule(ABC):
    def __init__(self, layer, percdamp=0.01, blocksize=128):
        self.layer = layer
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.dev = layer.weight.device

        self.W = layer.weight.data.clone()
        if hasattr(self.layer, "flatten") and isinstance(self.layer, torch.nn.Conv2d):
            self.W = self.W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            self.W = self.W.t()

        self.rows, self.columns = self.W.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 4:
            inp = inp[0, 0, :, :]
        inp = inp.squeeze()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t().float()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def prepare_weight_and_hessian(self):
        W = self.W.float()
        H = self.H
        if torch.isnan(H).any():
            err_msg = "Hessian contains NaN values."
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.H.detach().cpu()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        return W, H

    def compute_hinv(self, H, percdamp):
        while percdamp < 1.0:
            try:
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp
                if bh.name == "npu":
                    H_cpu = H.to("cpu")
                    H_inv = torch.linalg.cholesky(H_cpu)
                    H_inv = torch.cholesky_inverse(H_inv)
                    H_inv = torch.linalg.cholesky(H_inv, upper=True)
                    return H_inv.to(self.dev)

                H_inv = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(H_inv)
                H_inv = torch.linalg.cholesky(H_inv, upper=True)
                return H_inv
            except torch._C._LinAlgError:
                percdamp += 0.01
                logger.warning(
                    f"Cholesky failed with percdamp={percdamp-0.01:.4f}, retrying with percdamp={percdamp:.4f}..."
                )
        raise RuntimeError("Cholesky failed.")

    @abstractmethod
    def process(self): ...

    def print_log(self, losses, q_weight):
        avg_hessian_err = torch.sum(losses).item() / self.nsamples
        
        with torch.no_grad():
            diff = q_weight.reshape(self.layer.weight.shape) - self.layer.weight.data
            norm_loss = torch.norm(diff.float()).item()

        w_in, w_out = q_weight.shape[1], q_weight.shape[0]
        label_width = 26
        logger.info(f"{'Layer Shape:':<{label_width}} [In={w_in}, Out={w_out}]")
        logger.info(f"{'Hessian Error (sample):':<{label_width}} {avg_hessian_err:.4e}")
        logger.info(f"{'Weight Norm Loss (L2):':<{label_width}} {norm_loss:.4e}")

    def write_back(self, q_weight):
        if isinstance(self.layer, transformers.Conv1D):
            q_weight = q_weight.t()
        self.layer.weight.data.copy_(
            q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        )

    def free(self):
        self.H = None
        self.W = None
        bh.empty_cache()
