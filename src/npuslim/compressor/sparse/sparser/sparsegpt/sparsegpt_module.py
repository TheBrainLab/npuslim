from loguru import logger
import torch
import transformers
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule


class SparseGPTModule(BaseHessianModule):

    def fasterprune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01):
        is_conv1d = isinstance(self.layer, transformers.Conv1D)

        def prune_block_op(W1, Hinv1, i1, i2):
            count = i2 - i1
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            if prunen == 0:
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q

                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            return Q1, Losses1, Err1, W1

        W_final, Losses_final = self.faster_compress_common(
            blocksize, percdamp, prune_block_op
        )

        # Compute loss
        total_loss = torch.sum(Losses_final).item()
        with torch.no_grad():
            W_orig = self.layer.weight.data.float()
            if isinstance(self.layer, transformers.Conv1D):
                W_orig = W_orig.t()
            weight_rmse = torch.sqrt(torch.mean((W_final - W_orig) ** 2)).item()
            relative_diff = (torch.norm(W_final - W_orig) / torch.norm(W_orig)).item()

        full_name = getattr(self, "layer_name", "Unknown")
        logger.info(
            f"Module: {full_name: <20} | "
            f"Hessian Error: {total_loss:.4e} | "
            f"Weight RMSE: {weight_rmse:.4e} | "
            f"Relative Diff: {relative_diff:.2%}"
        )

        if is_conv1d:
            W_final = W_final.t()
        self.layer.weight.data = W_final.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
