import torch
from ...core.quant_func import compute_scales_with_zero
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule


class GPTQModule(BaseHessianModule):
    def fasterquant(
        self, blocksize=128, percdamp=0.01, group_size=-1, actorder=True, sym=True
    ):
        W = self.W_orig.clone()
        H = self.H.clone()

        scales, zeros = [], []
        if group_size != -1:
            for i in range(0, self.columns, group_size):
                s, z = compute_scales_with_zero(
                    W[:, i : i + group_size], bits=4, sym=sym
                )
                scales.append(s)
                zeros.append(z)

        perm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Hinv, dead = self._prepare_hinv(H, percdamp)
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, i1:i2].clone()
            Hinv1 = Hinv[i1:i2, i1:i2]
            Err1 = torch.zeros_like(W1)

            for i in range(i2 - i1):
                w = W1[:, i]
                d = Hinv1[i, i]

                idx = i1 + i
                if actorder:
                    idx = perm[idx]
                weight_scale = scales[idx // group_size]
                weight_zero = zeros[idx // group_size]

                q = torch.clamp(
                    torch.round(w.unsqueeze(1) / weight_scale) + weight_zero, 0, 15
                )
                q = weight_scale * (q - weight_zero)
                q = q.flatten()

                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                W[:, i1 + i] = q
                Err1[:, i] = err

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            W = W[:, invperm]
            g_idx = [perm[i] // group_size for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, device=self.dev)[invperm]
        else:
            g_idx = torch.tensor(
                [i // group_size for i in range(self.columns)], device=self.dev
            )

        self.layer.weight.data.copy_(W.reshape(self.layer.weight.shape))
        return torch.cat(scales, dim=1), torch.cat(zeros, dim=1), g_idx
