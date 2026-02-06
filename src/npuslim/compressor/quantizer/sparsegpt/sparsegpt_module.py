import torch
import torch.nn as nn
import transformers

from npuslim.compressor.core.base_hessian_module import BaseHessianModule


class SparseGPTModule(BaseHessianModule):
    def __init__(self, layer, config=None):
        super().__init__(layer, config=config)
        self.sparsity = getattr(self.config, "sparsity", 0.5)
        self.prunen = getattr(self.config, "prunen", 0)
        self.prunem = getattr(self.config, "prunem", 0)
        self.blocksize = getattr(self.config, "blocksize", 128)

    def fasterprune(self, **kwargs):
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        W_orig = W.clone()
        H = self.H
        del self.H

        Hinv = self.compute_hinv(H)
        Losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if self.prunen == 0:
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * self.sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if self.prunen != 0 and i % self.prunem == 0:
                    tmp = (
                        W1[:, i : (i + self.prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + self.prunem)].reshape((1, -1)))
                        ** 2
                    )
                    mask1.scatter_(
                        1,
                        i + torch.topk(tmp, self.prunen, dim=1, largest=False)[1],
                        True,
                    )

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        avg_loss = torch.sum(Losses).item() / self.nsamples
        norm_loss = torch.norm(W - W_orig).item()
        self.print_log(avg_loss, norm_loss)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        self.postproc()
        self.free()
