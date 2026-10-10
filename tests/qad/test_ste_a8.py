"""Track A 组合闸门测试：A8 激活 per-token 动态量化语义（单一语义源）。

覆盖 §6.8.3 步骤 2 / §6.9.2#4：训练/评测/部署三方必须共用
npuslim.qad.ste.activation_dynamic_int8，本测试锁定其语义性质。
"""

import torch

from npuslim.qad.ste import activation_dynamic_int8


class TestActivationDynamicInt8:
    def test_row_absmax_maps_to_grid_edge(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(6, 32, generator=g) * 5.0
        y = activation_dynamic_int8(x)
        s = x.abs().amax(dim=-1, keepdim=True) / 127.0
        assert torch.allclose(y.abs().amax(dim=-1, keepdim=True), s * 127, rtol=1e-5)

    def test_values_lie_on_scale_grid(self):
        x = torch.randn(4, 16) * torch.tensor([[10.0], [1.0], [100.0], [0.01]])
        y = activation_dynamic_int8(x)
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        assert torch.allclose(y / s, (y / s).round(), atol=1e-4)

    def test_symmetric_clamp_bounds(self):
        x = torch.tensor([[1e3, -1e3, 1.0], [0.0, 2.0, -3.0]])
        y = activation_dynamic_int8(x)
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        assert (y.abs() <= 128.0 * s + 1e-6).all()

    def test_idempotent(self):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(5, 24, generator=g) * 3.0
        y = activation_dynamic_int8(x)
        assert torch.equal(activation_dynamic_int8(y), y)

    def test_zero_row_stays_zero_without_nan(self):
        x = torch.zeros(3, 8)
        y = activation_dynamic_int8(x)
        assert torch.isfinite(y).all()
        assert (y == 0).all()

    def test_hand_computed_reference(self):
        x = torch.tensor([[0.0, 127.0], [1.0, -2.0]])
        y = activation_dynamic_int8(x, bits=8)
        s0 = torch.tensor(127.0 / 127.0)
        s1 = torch.tensor(2.0 / 127.0)
        expected = torch.tensor([[0.0 * s0, 127.0 * s0], [64.0 * s1, -127.0 * s1]])
        assert torch.allclose(y, expected, atol=1e-5)

    def test_matches_eval_alias(self):
        from npuslim.qad.eval_rotated_ppl import fake_quant_act_per_token

        g = torch.Generator().manual_seed(11)
        x = torch.randn(3, 12, generator=g) * 8.0
        assert torch.equal(activation_dynamic_int8(x), fake_quant_act_per_token(x))
