"""CAKLD 损失单测（任务 1，§6.9.2#1 / §5.14-B3）。

覆盖：
- cakld 语义：conf·rev + (1−conf)·fwd（AngelSlim loss.py cakld 同构）
- conf 取值：label ∈ top-K → exp(log_p)；label ∈ 尾部 → rest/(V-K)
- 边界一致：conf≡1 → 等价 reverse.mean()；conf≡0 → 等价 forward.mean()
- 回归：kl_distill_loss 重构后 forward/reverse 与独立手算公式一致
"""

import math

import torch

from npuslim.qad.kl_loss import cakld_distill_loss, kl_distill_loss

V, K, B = 10, 3, 2


def _fixture():
    torch.manual_seed(1)
    sl = torch.randn(B, V)
    t_ids = torch.tensor([[1, 3, 5], [2, 7, 0]])
    t_lp = torch.log_softmax(torch.randn(B, V), dim=-1).gather(-1, t_ids)
    labels = torch.tensor([3, 0])   # 行 0：label 在 top-K（位 1）；行 1：label=0 在 top-K（位 2）
    return sl, t_ids, t_lp, labels


def _ref_fwd_rev(sl, t_ids, t_lp, V, tau=1.0):
    """独立手算参考：docstring 公式逐行实现（不调用被测代码）。"""
    z = sl.float() / tau
    lse = torch.logsumexp(z, dim=-1)
    s_lp = z.gather(-1, t_ids) - lse.unsqueeze(-1)
    log_p = t_lp.float()
    p = torch.exp(log_p)
    rest = (1.0 - p.sum(dim=-1)).clamp_min(0.0)
    log_rest = torch.log(rest.clamp_min(1e-12)) - math.log(V - t_ids.shape[-1])
    q = torch.exp(s_lp)
    q_sum = q.sum(dim=-1).clamp_max(1.0 - 1e-7)
    q_rest = 1.0 - q_sum
    log_q_rest = torch.log(q_rest.clamp_min(1e-12))
    fwd = (p * (log_p - s_lp)).sum(dim=-1) + rest * (log_rest - log_q_rest)
    rev = (q * (s_lp - log_p)).sum(dim=-1) + q_rest * (log_q_rest - log_rest)
    return fwd, rev


class TestCakld:
    def test_matches_manual_mixture(self):
        """Given: 固定 teacher/student/label。When: cakld。Then: = conf·rev + (1−conf)·fwd（手算）。"""
        sl, t_ids, t_lp, labels = _fixture()
        fwd, rev = _ref_fwd_rev(sl, t_ids, t_lp, V)
        # conf：行 0 label=3 ∈ top-K 位 1 → exp(t_lp[0,1])；行 1 label=0 ∈ top-K 位 2 → exp(t_lp[1,2])
        conf = torch.tensor([math.exp(t_lp[0, 1].item()), math.exp(t_lp[1, 2].item())])
        expect = (conf * rev + (1 - conf) * fwd).mean()
        got = cakld_distill_loss(sl, t_ids, t_lp, labels, V)
        assert torch.allclose(got, expect, atol=1e-6)

    def test_tail_label_gets_tail_conf(self):
        """Given: label 不在 top-K。When: cakld。Then: conf = rest/(V-K)（均匀尾）。"""
        sl, t_ids, t_lp, _ = _fixture()
        labels = torch.tensor([9, 9])   # 9 不在任一行 top-K
        p = torch.exp(t_lp.float())
        rest = (1.0 - p.sum(dim=-1)).clamp_min(0.0) / (V - K)
        fwd, rev = _ref_fwd_rev(sl, t_ids, t_lp, V)
        expect = (rest * rev + (1 - rest) * fwd).mean()
        got = cakld_distill_loss(sl, t_ids, t_lp, labels, V)
        assert torch.allclose(got, expect, atol=1e-6)

    def test_conf_one_reduces_to_reverse(self):
        """Given: teacher 把全部概率放在 label 上（label=top-1，log_p=0）。When: cakld。Then: = reverse.mean()。"""
        sl = torch.randn(B, V)
        labels = torch.tensor([4, 4])
        t_ids = torch.tensor([[4, 0, 1], [4, 2, 3]])
        t_lp = torch.tensor([[0.0, -15.0, -15.0], [0.0, -15.0, -15.0]])
        got = cakld_distill_loss(sl, t_ids, t_lp, labels, V)
        _, rev = _ref_fwd_rev(sl, t_ids, t_lp, V)
        assert torch.allclose(got, rev.mean(), atol=1e-5)

    def test_conf_zero_reduces_to_forward(self):
        """Given: teacher 对 label 置信度为 0（label 不在 top-K 且 rest→0 不可构造，改用
        label ∈ top-K 但概率极低：t_lp = -15）。When: cakld。Then: ≈ forward.mean()。"""
        sl = torch.randn(B, V)
        labels = torch.tensor([4, 4])
        t_ids = torch.tensor([[0, 1, 4], [2, 3, 4]])
        t_lp = torch.tensor([[0.0, 0.0, -15.0], [0.0, 0.0, -15.0]])
        got = cakld_distill_loss(sl, t_ids, t_lp, labels, V)
        fwd, _ = _ref_fwd_rev(sl, t_ids, t_lp, V)
        assert torch.allclose(got, fwd.mean(), atol=1e-5)


class TestKlDistillLossRegression:
    def test_forward_matches_independent_formula(self):
        """重构后 kl_distill_loss（forward）与独立手算公式逐值一致。"""
        sl, t_ids, t_lp, _ = _fixture()
        fwd, _ = _ref_fwd_rev(sl, t_ids, t_lp, V)
        got = kl_distill_loss(sl, t_ids, t_lp, V, mode="forward")
        assert torch.allclose(got, fwd.mean(), atol=1e-6)

    def test_reverse_matches_independent_formula(self):
        sl, t_ids, t_lp, _ = _fixture()
        _, rev = _ref_fwd_rev(sl, t_ids, t_lp, V)
        got = kl_distill_loss(sl, t_ids, t_lp, V, mode="reverse")
        assert torch.allclose(got, rev.mean(), atol=1e-6)
