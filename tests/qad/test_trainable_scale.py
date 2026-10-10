"""任务 2（§5.14-B1 scale-only 对照臂）：可训练 scale 的 LSQ 式梯度与优化收敛单测。

覆盖：
- STE backward 的 scale 梯度流（LSQ 约定：∂(v·s)/∂s ≈ v）
- 逐行 scale 的形状与逐元素正确性
- 优化 scale 可降低权重量化 MSE（scale-only 臂的价值前提）
"""

import torch

from npuslim.qad.ste import compute_weight_scale, fake_quant_weight


class TestTrainableScale:
    def test_scale_grad_flows_lsq_style(self):
        """Given: scale 为叶 Parameter。When: fake_quant 对 sum 输出反向。Then: scale 梯度
        = Σ v（LSQ 式），形状 [rows,1]。"""
        torch.manual_seed(2)
        w = torch.randn(4, 8)
        s = torch.nn.Parameter(compute_weight_scale(w, 4).clone())
        y = fake_quant_weight(w, bits=4, scale=s)
        y.sum().backward()
        v = torch.clamp(torch.round(w / s), -8, 7)
        assert s.grad is not None
        assert s.grad.shape == (4, 1)
        assert torch.allclose(s.grad, v.sum(dim=-1, keepdim=True), atol=1e-5)

    def test_scale_optimization_reduces_quant_mse(self):
        """Given: minmax 初始 scale。When: Adam 优化 40 步。Then: 权重量化 MSE 下降。"""
        torch.manual_seed(4)
        w = torch.randn(6, 8) * 3.0
        s = torch.nn.Parameter((w.abs().amax(-1, keepdim=True) / 7.0).detach().clone())
        opt = torch.optim.Adam([s], lr=0.05)
        first = None
        last = None
        for _ in range(40):
            y = fake_quant_weight(w, bits=4, scale=s)
            loss = (y - w).pow(2).mean()
            if first is None:
                first = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        assert last < first

    def test_frozen_scale_has_no_grad(self):
        """Given: scale 为普通张量（历史行为）。When: fake_quant 反向。Then: grad_scale
        为 None（w 恒等直通语义不变）。"""
        w = torch.randn(3, 8, requires_grad=True)
        s = (w.abs().amax(-1, keepdim=True) / 7.0).detach()
        y = fake_quant_weight(w, bits=4, scale=s)
        y.sum().backward()
        assert w.grad is not None
        assert torch.allclose(w.grad, torch.ones_like(w))
