from abc import ABC, abstractmethod
from loguru import logger
import torch
import math
import transformers
from npuslim.utils.backend import bh


class BaseHessianModule(ABC):
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device

        self.W = layer.weight.data
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

    def _compute_hinv(self, H, percdamp):
        while percdamp < 1.0:
            try:
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp
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

    def write_back(self, W):
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data.copy_(
            W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        )

    def free(self):
        self.H = None
        bh.empty_cache()

    def _execute_obs_process(self, blocksize=128, percdamp=0.01, actorder=False):
        """
        抽象的 OBS 核心执行框架
        """
        # 1. 准备 Hessian 逆矩阵
        # H = self.H.clone()
        H = self.H
        if torch.isnan(H).any():
            err_msg = "Hessian contains NaN values."
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.H.detach().cpu()
        del self.H

        W = self.W.clone().float()
        self._current_W_ref = W  # 让子类能访问到 W 的实时状态
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        perm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Hinv = self._compute_hinv(H, percdamp)
        total_hessian_loss = 0.0
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, i1:i2].clone()
            Hinv1 = Hinv[i1:i2, i1:i2]
            Err1 = torch.zeros_like(W1)
            
            if actorder:
                block_idxs = perm[i1:i2]
            else:
                block_idxs = torch.arange(i1, i2, device=self.dev)
            self.on_block_start(W1, Hinv1, block_idxs)

            for i in range(i2 - i1):
                w = W1[:, i]
                d = Hinv1[i, i]
                current_idx = block_idxs[i].item()
                q = self.transform_column(w, d, current_idx)
                
                # --- [compute Hessian Loss] ---
                col_hessian_loss = torch.sum((w - q) ** 2) / (d ** 2)
                total_hessian_loss += col_hessian_loss.item()

                # --- update weight ---
                err1 = (w - q) / d
                Err1[:, i] = err1
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

            W[:, i1:i2] = W1
            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            invperm = torch.argsort(perm)
            W = W[:, invperm]
        
        # --- [compute Frobenius Norm Loss] ---
        with torch.no_grad():
            weight_norm_loss = torch.norm(W - self.W).item()
            relative_diff = weight_norm_loss / torch.norm(self.W).item()

        logger.info(f"Hessian Loss (Approx Error): {total_hessian_loss / 2:.4e}")
        logger.info(f"Norm Loss (Weight Diff): {weight_norm_loss:.4f} (Relative: {relative_diff:.2%})")
        return W

    @abstractmethod
    def on_block_start(self, W1, Hinv1, block_idxs):
        """
        块处理前的钩子函数。

        Args:
            W1 (Tensor): 当前块的权重子矩阵，维度为 [Rows, BlockSize]。
                        它是原始权重 W 在列范围 [i1:i2] 的视图或副本。
            Hinv1 (Tensor): 当前块对应的 Hessian 逆矩阵块，维度为 [BlockSize, BlockSize]。
                        它决定了块内权重的相互影响关系.
            block_idxs (Tensor): 当前块中列的全局索引，维度为 [BlockSize]。
                                当 actorder 启用时，它用于定位量化参数 (scale/zero) 或判断组索引。
           
        """
        raise NotImplementedError

    @abstractmethod
    def transform_column(self, w, d, col_idx):
        """
        定义单列权重的修改逻辑（剪枝或量化）。

        Args:
            w (Tensor): 当前正在处理的列权重，维度为 [Rows].
                    注意：这是已经根据前面列的误差更新过的“最新”权重。
            d (float): 当前列在 Hessian 逆矩阵对角线上的值 (Hinv[i, i])。
                    在 OBS 公式中, 它是误差分母, 用于计算误差传播系数: err = (w - q) / d。
            col_idx (int): 当前列在全局权重矩阵中的索引。
                        常用于 actorder 开启时定位量化参数 (scale/zero) 或判断组索引。

        Returns:
            q (Tensor): 修改后的列权重 (如量化后的值或剪枝后的 0), 维度为 [Rows]。
        """
        raise NotImplementedError
