"""routing top-k 重合率探针单测（任务 3，§6.9.8-P4）。"""
import torch
from npuslim.qad.routing_probe import topk_overlap, routing_overlap_per_layer, capture_router_topk


def test_identical_routing_full_overlap():
    g = torch.randn(6, 8)
    t = capture_router_topk(g, 2)
    assert (topk_overlap(t, t) == 1.0).all()


def test_disjoint_routing_zero_overlap():
    a = torch.tensor([[0, 1], [2, 3]])
    b = torch.tensor([[4, 5], [6, 7]])
    assert (topk_overlap(a, b) == 0.0).all()


def test_partial_overlap_value():
    a = torch.tensor([[0, 1, 2]])
    b = torch.tensor([[0, 1, 9]])
    assert abs(topk_overlap(a, b).item() - 2.0 / 3.0) < 1e-6


def test_order_insensitive():
    a = torch.tensor([[3, 7]])
    b = torch.tensor([[7, 3]])
    assert (topk_overlap(a, b) == 1.0).all()


def test_per_layer_mean():
    a = {0: torch.tensor([[0, 1]]), 1: torch.tensor([[2, 3]])}
    b = {0: torch.tensor([[0, 1]]), 1: torch.tensor([[2, 9]])}
    r = routing_overlap_per_layer(a, b)
    assert r[0] == 1.0 and abs(r[1] - 0.5) < 1e-9
