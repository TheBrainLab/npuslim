import torch
import math
from torch import nn


class BaseHessianModule:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        rows = layer.weight.shape[0]
        cols = layer.weight.shape[1]
        self.rows = rows
        self.columns = cols
        self.nsamples = 0
        self.H = torch.zeros((cols, cols), device=self.dev)

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
    
    def _prepare_hinv(self, percdamp):
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # TODO: not supported in NPU
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H).mH 
        return H, dead

    def faster_compress_common(self, blocksize, percdamp, compress_op):
        """
        通用的 OBC 压缩框架。
        compress_op: 一个回调函数，负责具体 block 内的 Q 计算逻辑。
        """
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        Hinv, dead = self._prepare_hinv(percdamp)
        W[:, dead] = 0
        
        Losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            Q1, Losses1, Err1, W1_updated = compress_op(W1, Hinv1, i1, i2)
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return W, Losses

    def free(self):
        self.H = None
        torch.npu.empty_cache()