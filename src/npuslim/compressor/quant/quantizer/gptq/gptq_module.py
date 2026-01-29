import torch
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule
from npuslim.compressor.quant.core.quant_func import compute_scales_with_zero


class GPTQModule(BaseHessianModule):
    def __init__(
        self,
        layer,
        quant_bits=4,
        group_size=-1,
        sym=True,
        actorder=True,
        static_groups=True,
        **kwargs,
    ):
        super().__init__(layer, **kwargs)
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.sym = sym
        self.actorder = actorder
        self.static_groups = static_groups

    def process(self):
        W, H = self.prepare_weight_and_hessian()

        g_idx = []
        scale = []
        zero = []
        now_idx = 1
        if self.static_groups:
            for i in range(0, self.columns, self.group_size):
                weight_scale, weight_zero = compute_scales_with_zero(
                    W[:, i : (i + self.group_size)], bits=self.quant_bits, sym=self.sym
                )
                scale.append(weight_scale)
                zero.append(weight_zero)
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        
        losses = torch.zeros_like(W)
        q_weight = torch.zeros_like(W)
        Hinv = self.compute_hinv(H, self.percdamp)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            w1 = W[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = w1[:, i]
                d = hinv1[i, i]

                if self.group_size != -1:
                    if not self.static_groups:
                        if (i1 + i) % self.group_size == 0:
                            weight_scale, weight_zero = compute_scales_with_zero(
                                W[:, (i1 + i) : (i1 + i + self.group_size)],
                                bits=self.quant_bits,
                                sym=self.sym,
                            )

                        if ((i1 + i) // self.group_size) - now_idx == -1:
                            scale.append(weight_scale)
                            zero.append(weight_zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if self.actorder:
                            idx = perm[idx]
                        weight_scale = scale[idx // self.group_size]
                        weight_zero = zero[idx // self.group_size]

                maxq = torch.tensor(2**self.quant_bits - 1)
                q = torch.clamp(
                    torch.round(w.unsqueeze(1) / weight_scale) + weight_zero, 0, maxq
                )
                q = weight_scale * (q - weight_zero)
                q = q.flatten()
                q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2

                err = (w - q) / d
                w1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = err

            q_weight[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2

            W[:, i2:] -= err1.matmul(Hinv[i1:i2, i2:])

        group_size = self.group_size if self.group_size != -1 else self.columns
        if self.static_groups and self.actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=q_weight.device)
        if self.actorder:
            q_weight = q_weight[:, invperm]
            g_idx = g_idx[invperm]

        self.print_log(losses=losses, q_weight=q_weight)
        self.write_back(q_weight=q_weight)

        if scale == []:
            scale = weight_scale
            zero = torch.zeros_like(weight_scale)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        losses = losses.cpu()
        q_weight = q_weight.cpu()
        W = W.cpu()
        H = H.cpu()
        Hinv = Hinv.cpu()
        del losses, q_weight, W, H, Hinv
        self.W = self.W.cpu()
        del self.W
        
        self.free()
        return scale, zero, g_idx
