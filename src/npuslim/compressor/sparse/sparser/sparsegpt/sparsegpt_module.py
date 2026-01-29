import torch
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule


class SparseGPTModule(BaseHessianModule):
    def __init__(self, layer, sparsity=0, prunen=0, prunem=0, **kwargs):
        super().__init__(layer, **kwargs)
        self.sparsity = sparsity
        self.prunen = prunen
        self.prunem = prunem

    def process(self):
        W, H = self.prepare_weight_and_hessian()

        losses = torch.zeros(self.rows, device=self.dev)
        mask = None
        Hinv = self.compute_hinv(H, self.percdamp)
        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if self.prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][
                        int(tmp.numel() * self.sparsity)
                    ]
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
                losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            W[:, i1:i2] = Q1
            losses += torch.sum(losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        self.print_log(losses=losses, q_weight=W)
        self.write_back(q_weight=W)

        losses = losses.cpu()
        W = W.cpu()
        H = H.cpu()
        Hinv = Hinv.cpu()
        del losses, W, H, Hinv
        self.W = self.W.cpu()
        del self.W
        
        self.free()
