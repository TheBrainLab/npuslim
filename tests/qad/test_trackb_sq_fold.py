"""Track B：Path2 SQ 折叠原语 fp 恒等对拍（tiny，§6.8.3 步骤 2 / §6.9.2 Track B）。

闸门标准（E3 同款）：折叠前后网络函数恒等——fp32 atol 1e-5、bf16 atol 1e-2 量级。
原语直接测（object.__new__ 绕过算法类 __init__，零 pipeline 改动）：
  _sq_fold_pair  @staticmethod：consumer 输入列 ×s；producer 输出行与 bias ÷s
  _sq_fold_ov    实例方法（bare instance + 注入 _runtime_model/_TAG）：
                 MLA v 通道头结构折叠（o_proj 输入列 ×s；kv_b_proj v 输出行 ÷s）
"""

import types

import torch
import torch.nn as nn

from npuslim.algorithms.quantization.gptq.gptq_algo import GPTQAlgorithm


def _bare_algo():
    inst = object.__new__(GPTQAlgorithm)
    inst._TAG = "TRACKB"
    inst._runtime_model = types.SimpleNamespace(
        config=types.SimpleNamespace(v_head_dim=3, qk_nope_head_dim=4,
                                     num_attention_heads=2))
    return inst


class TestSqFoldPair:
    def test_preserves_function_with_bias(self):
        torch.manual_seed(3)
        prod = nn.Linear(16, 24)
        cons = nn.Linear(24, 12, bias=False)
        s = torch.rand(24) * 1.5 + 0.5
        x = torch.randn(5, 16)
        y_pre = cons(prod(x))
        GPTQAlgorithm._sq_fold_pair(cons, prod, s)
        y_post = cons(prod(x))
        assert torch.allclose(y_pre, y_post, atol=1e-4)

    def test_preserves_function_without_bias(self):
        torch.manual_seed(5)
        prod = nn.Linear(16, 24, bias=False)
        cons = nn.Linear(24, 12, bias=False)
        s = torch.rand(24) * 1.5 + 0.5
        x = torch.randn(7, 16)
        y_pre = cons(prod(x))
        GPTQAlgorithm._sq_fold_pair(cons, prod, s)
        y_post = cons(prod(x))
        assert torch.allclose(y_pre, y_post, atol=1e-4)

    def test_weights_scaled_as_documented(self):
        torch.manual_seed(6)
        prod = nn.Linear(8, 6)
        cons = nn.Linear(6, 4, bias=False)
        b_prod = prod.bias.detach().clone()
        w_prod, w_cons = prod.weight.detach().clone(), cons.weight.detach().clone()
        s = torch.tensor([2.0, 1.0, 0.5, 1.0, 4.0, 0.25])
        GPTQAlgorithm._sq_fold_pair(cons, prod, s)
        assert torch.allclose(cons.weight.detach(), w_cons * s.view(1, -1), atol=1e-6)
        assert torch.allclose(prod.weight.detach(), w_prod / s.view(-1, 1), atol=1e-6)
        assert torch.allclose(prod.bias.detach(), b_prod / s, atol=1e-6)


class TestSqFoldOv:
    @staticmethod
    def _tiny_mla():
        nh, qk_nope, v_head, hidden = 2, 4, 3, 8
        kvb = nn.Linear(hidden, nh * (qk_nope + v_head))
        o = nn.Linear(nh * v_head, hidden, bias=False)
        return nh, qk_nope, v_head, hidden, kvb, o

    @staticmethod
    def _v_flat(kv_out, nh, v_head):
        per = kv_out.shape[-1] // nh
        v = kv_out.view(*kv_out.shape[:-1], nh, per)[..., -v_head:]
        return v.reshape(*kv_out.shape[:-1], nh * v_head)

    def test_ov_fold_preserves_v_path(self):
        torch.manual_seed(9)
        nh, qk_nope, v_head, hidden = 2, 4, 3, 8
        _, _, _, _, kvb, o = self._tiny_mla()
        s = torch.rand(nh * v_head) * 1.5 + 0.5
        x = torch.randn(6, hidden)

        kv_out_pre = kvb(x)
        v_pre = o(self._v_flat(kv_out_pre, nh, v_head))
        qk_part_pre = kv_out_pre.view(*kv_out_pre.shape[:-1], nh, -1)[..., :-v_head]

        inst = _bare_algo()
        inst._sq_fold_ov(None, o, kvb, s, num_heads=nh, kv_heads=1, v_path="", o_path="")

        kv_out_post = kvb(x)
        v_post = o(self._v_flat(kv_out_post, nh, v_head))
        qk_part_post = kv_out_post.view(*kv_out_post.shape[:-1], nh, -1)[..., :-v_head]

        assert torch.allclose(v_pre, v_post, atol=1e-4)
        assert torch.allclose(qk_part_pre, qk_part_post, atol=1e-4)

    def test_ov_fold_weights_scaled_as_documented(self):
        torch.manual_seed(12)
        nh, qk_nope, v_head, hidden = 2, 4, 3, 8
        _, _, _, _, kvb, o = TestSqFoldOv._tiny_mla_parts()
        s = torch.rand(nh * v_head) * 1.5 + 0.5
        w_o, w_kvb = o.weight.detach().clone(), kvb.weight.detach().clone()

        inst = _bare_algo()
        inst._sq_fold_ov(None, o, kvb, s, num_heads=nh, kv_heads=1, v_path="", o_path="")

        assert torch.allclose(o.weight.detach(), w_o * s.view(1, -1), atol=1e-5)
        w = w_kvb.view(nh, qk_nope + v_head, hidden)
        w_new = kvb.weight.detach().view(nh, qk_nope + v_head, hidden)
        assert torch.allclose(w_new[:, -v_head:, :], w[:, -v_head:, :] / s.view(nh, v_head, 1), atol=1e-5)
        assert torch.allclose(w_new[:, :-v_head, :], w[:, :-v_head, :], atol=1e-6)

    @staticmethod
    def _tiny_mla_parts():
        torch.manual_seed(12)
        nh, qk_nope, v_head, hidden = 2, 4, 3, 8
        kvb = nn.Linear(hidden, nh * (qk_nope + v_head))
        o = nn.Linear(nh * v_head, hidden, bias=False)
        return nh, qk_nope, v_head, hidden, kvb, o


class TestSqFoldOvBf16:
    def test_ov_fold_bf16_within_e3_gate(self):
        """bf16 口径：折叠舍入后的函数偏差 ≤ 闸门（bf16 舍入界，atol 1e-2）。"""
        torch.manual_seed(15)
        nh, qk_nope, v_head, hidden = 2, 4, 3, 8
        kvb = nn.Linear(hidden, nh * (qk_nope + v_head)).to(torch.bfloat16)
        o = nn.Linear(nh * v_head, hidden, bias=False).to(torch.bfloat16)
        s = (torch.rand(nh * v_head) * 1.5 + 0.5).to(torch.bfloat16)
        x = (torch.randn(6, hidden, generator=torch.Generator().manual_seed(2)) * 2).to(torch.bfloat16)

        v_pre = o(TestSqFoldOv._v_flat(kvb(x), nh, v_head))
        inst = _bare_algo()
        inst._sq_fold_ov(None, o, kvb, s, num_heads=nh, kv_heads=1, v_path="", o_path="")
        v_post = o(TestSqFoldOv._v_flat(kvb(x), nh, v_head))

        assert torch.allclose(v_pre.float(), v_post.float(), atol=1e-2)
