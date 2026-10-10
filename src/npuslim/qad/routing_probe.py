"""routing top-k 重合率探针（任务 3，§6.9.8-P4 / §6.9.2#6）。

量化误差 → 路由漂移 → 行为退化证据链的第一环：同一输入下，量化学生与
BF16 teacher 的专家选择一致性。MoE 独有、近乎免费（router 前向顺手捕获）。

用法（真实模型，等 4 节点窗口）：
  两模型各挂 capture 钩子 → 同批输入前向 → 逐层 top-k 集合 Jaccard/重合数。
本模块只提供纯函数（可 CPU 单测）：
  routing_overlap(topk_a, topk_b) → (mean_overlap, per_layer)
"""

from __future__ import annotations

import torch


def topk_overlap(topk_a: torch.Tensor, topk_b: torch.Tensor) -> torch.Tensor:
    """逐样本 top-k 专家集合重合数。

    Args:
        topk_a: [B, k] 学生（或对照）侧的专家 id
        topk_b: [B, k] teacher（或基线）侧的专家 id
    Returns:
        [B] 每样本重合率（0..1 = 重合数/k）
    """
    assert topk_a.shape == topk_b.shape, f"shape mismatch: {topk_a.shape} vs {topk_b.shape}"
    k = topk_a.shape[-1]
    match = (topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2)).any(dim=-1)   # [B, k_a]
    return match.sum(dim=-1).float() / k


def routing_overlap_per_layer(topk_by_layer_a: dict, topk_by_layer_b: dict) -> dict:
    """逐层重合率：{layer_idx: [B] 重合数} 两输入的交集键上求均值。"""
    out = {}
    for layer in sorted(set(topk_by_layer_a) & set(topk_by_layer_b)):
        out[layer] = topk_overlap(topk_by_layer_a[layer], topk_by_layer_b[layer]).mean().item()
    return out


def capture_router_topk(gate_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """router gate logits [B, E] → top-k 专家 id [B, k]（与部署 top-k 语义一致）。"""
    return torch.topk(gate_logits, top_k, dim=-1).indices


class TestRoutingOverlap:
    def test_identical_routing_full_overlap(self):
        g = torch.randn(6, 8)
        t = capture_router_topk(g, 2)
        assert (topk_overlap(t, t) == 2.0).all()

    def test_disjoint_routing_zero_overlap(self):
        a = torch.tensor([[0, 1], [2, 3]])
        b = torch.tensor([[4, 5], [6, 7]])
        assert (topk_overlap(a, b) == 0.0).all()

    def test_partial_overlap_value(self):
        a = torch.tensor([[0, 1, 2]])
        b = torch.tensor([[0, 1, 9]])
        assert topk_overlap(a, b).item() == 2.0 / 3.0

    def test_per_layer_mean(self):
        a = {0: torch.tensor([[0, 1]]), 1: torch.tensor([[2, 3]])}
        b = {0: torch.tensor([[0, 1]]), 1: torch.tensor([[2, 9]])}
        r = routing_overlap_per_layer(a, b)
        assert r[0] == 1.0 and abs(r[1] - 0.5) < 1e-9

    def test_order_insensitive(self):
        a = torch.tensor([[3, 7]])
        b = torch.tensor([[7, 3]])
        assert (topk_overlap(a, b) == 2.0).all()
