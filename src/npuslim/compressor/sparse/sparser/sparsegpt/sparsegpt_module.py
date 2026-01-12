import torch
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule


class SparseGPTModule(BaseHessianModule):
    def __init__(self, layer, sparsity=0, prunen=0, prunem=0):
        super().__init__(layer)
        self.sparsity = sparsity
        self.prunen = prunen
        self.prunem = prunem
        
        # 对应源码中的 mask1 和当前块的局部计数
        self.mask1 = None
        self._local_i = 0

    def on_block_start(self, W1, Hinv1, block_idxs):
        """
        对应源码中 block 循环开始处对 mask1 的初始化逻辑
        """
        self._local_i = 0
        
        if self.prunen == 0:
            # --- 源码逻辑: 非结构化剪枝 ---
            # tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            # thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * self.sparsity)]
            # mask1 = tmp <= thresh
            self.mask1 = tmp <= thresh
        else:
            # --- 源码逻辑: N:M 结构化剪枝初始化 ---
            # mask1 = torch.zeros_like(W1) == 1
            self.mask1 = torch.zeros_like(W1) == 1
        
        # 将 W1 和 Hinv1 暂存，供 N:M 在 transform_column 中使用
        self._current_W1 = W1
        self._current_Hinv1 = Hinv1

    def transform_column(self, w, d, col_idx):
        """
        对应源码中列循环内部的逻辑
        """
        i = self._local_i
        
        # --- 源码逻辑: N:M 结构化剪枝实时计算 ---
        if self.prunen != 0 and i % self.prunem == 0:
            # tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
            # 注意：这里的 W1 会随误差传播更新，完全符合源码
            W_slice = self._current_W1[:, i:(i + self.prunem)]
            H_diag_slice = torch.diag(self._current_Hinv1)[i:(i + self.prunem)].reshape((1, -1))
            tmp = W_slice ** 2 / (H_diag_slice ** 2)
            
            # mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
            indices = torch.topk(tmp, self.prunen, dim=1, largest=False)[1]
            self.mask1.scatter_(1, i + indices, True)

        # --- 源码逻辑: 应用 Mask ---
        q = w.clone()
        # q[mask1[:, i]] = 0
        q[self.mask1[:, i]] = 0
        
        self._local_i += 1
        return q

    def prune(self, blocksize=128, percdamp=0.01, actorder=False):
        """
        封装执行函数
        """
        # 调用基类的 OBS 核心逻辑
        W_after = self._execute_obs_process(
            blocksize=blocksize, 
            percdamp=percdamp, 
            actorder=actorder
        )
        # 写回权重
        self.write_back(W_after)
        self.free()