import torch
from npuslim.compressor.helper.base_hessian_module import BaseHessianModule
from npuslim.compressor.quant.core.quant_func import compute_scales_with_zero


class GPTQModule(BaseHessianModule):
    def __init__(self, layer, quant_bits=4, group_size=-1, sym=True, static_groups=True):
        super().__init__(layer)
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.sym = sym
        self.static_groups = static_groups
        
        self.maxq = torch.tensor(2**self.quant_bits - 1)
        
    def _precompute_static_params(self):
        """对应源码开头：if static_groups: ..."""
        g_size = self.group_size if self.group_size != -1 else self.columns
        for i in range(0, self.columns, g_size):
            W_group = self.W[:, i : (i + g_size)].float()
            weight_scale, weight_zero = compute_scales_with_zero(
                W_group, bits=self.quant_bits, sym=self.sym
            )
            self.scales.append(weight_scale)
            self.zeros.append(weight_zero)

    def on_block_start(self, W1, Hinv1, block_idxs):
        # 动态模式不需要在块开始时做特殊操作，逻辑全在 transform_column
        pass

    def transform_column(self, w, d, col_idx):
        """
        完美对齐源码内部循环逻辑：
        if group_size != -1:
            if not static_groups: ... (动态模式)
            else: ... (静态模式)
        """

        g_size = self.group_size if self.group_size != -1 else self.columns
        
        if self.group_size != -1:
            if not self.static_groups:
                # --- [动态模式逻辑] ---
                # 源码: if (i1 + i) % group_size == 0:
                # 在我们的框架中，i1 + i 对应的就是当前处理的局部计数，
                # 但要注意 actorder 开启时，动态组是基于“排序后”的顺序来切分的
                
                # 我们需要维护一个处理进度的计数器 (self._current_count)
                if self._current_count % g_size == 0:
                    # 获取当前及后面 g_size 列的权重（在 W1 或 W 中）
                    # 注意：源码这里用的是 w_weight，即排序后且被前面误差补偿过的权重
                    # 这里的 self._current_W_view 是基类正在迭代的权重矩阵副本
                    start = self._current_count
                    end = min(start + g_size, self.columns)
                    W_group = self._current_W_ref[:, start:end]
                    
                    weight_scale, weight_zero = compute_scales_with_zero(
                        W_group, bits=self.quant_bits, sym=self.sym
                    )
                    self.scales.append(weight_scale)
                    self.zeros.append(weight_zero)
                
                # 动态模式下，scale 总是取最后加进来的那一个
                weight_scale = self.scales[-1]
                weight_zero = self.zeros[-1]
            else:
                # --- [静态模式逻辑] ---
                # 源码: idx = perm[i1 + i]; weight_scale = scale[idx // group_size]
                # 因为 col_idx 就是物理索引（perm[idx]），直接计算即可
                g_idx = col_idx // g_size
                weight_scale = self.scales[g_idx]
                weight_zero = self.zeros[g_idx]
        else:
            # 无分组情况，使用全局单一 scale (通常也是预计算好的)
            weight_scale = self.scales[-1]
            weight_zero = self.zeros[-1]

        # --- [量化执行] ---
        q = torch.clamp(
            torch.round(w.unsqueeze(1) / weight_scale) + weight_zero, 
            0, self.maxq.to(self.dev)
        )
        q = weight_scale * (q - weight_zero)
        
        self._current_count += 1
        return q.flatten()

    def quantize(self, blocksize=128, percdamp=0.01, actorder=True):
        """
        GPTQ 量化执行入口，返回量化参数以便后续存储
        """
        # 初始化状态
        self.scales = []
        self.zeros = []
        self._current_count = 0
        
        # 1. 如果是静态组模式，先计算原始权重的 scale/zero
        if self.static_groups:
            self._precompute_static_params()

        # 2. 调用基类执行 OBS 核心过程（内部会处理 actorder 排序、误差传播等）
        W_final = self._execute_obs_process(
            blocksize=blocksize, 
            percdamp=percdamp, 
            actorder=actorder
        )
        
        # 3. 将量化后的权重写回层
        self.write_back(W_final)

        # 4. 计算 g_idx (对齐源码逻辑)
        # 源码逻辑：如果 static_groups 且 actorder，则根据 perm 来确定每个物理位置对应的组
        group_size = self.group_size if self.group_size != -1 else self.columns
        g_idx = torch.tensor(
            [i // group_size for i in range(self.columns)], 
            dtype=torch.int32, device=self.dev
        )

        # 5. 拼接结果并转 CPU (参考源码最后的清理操作)
        scale_tensor = torch.cat(self.scales, dim=1)
        zero_tensor = torch.cat(self.zeros, dim=1)

        # 6. 彻底释放显存
        self.free()
        
        # 返回三个值：scale, zero, g_idx
        return scale_tensor, zero_tensor, g_idx

    def free(self):
        """
        清理子类特有的显存占用，并调用基类清理 H 矩阵
        """
        # 1. 清理子类特有的量化参数列表
        self.scales = None
        self.zeros = None
        
        # 2. 清理可能存在的中间张量
        if hasattr(self, 'maxq'):
            del self.maxq
            
        # 3. 调用基类清理 self.H 和 self.W (如果基类定义了的话)
        super().free()