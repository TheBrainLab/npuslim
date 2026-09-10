"""Per-expert output-space LoRA（QAD Phase 1）。

结构仿 InstinctRazor `Agu/Bgu/Adn/Bdn`（per-expert 四矩阵，gate_up/down 各一对）：
- gate_up: A [rank, hidden], B [2*inter, rank]（fused gate_up）
- down:    A [rank, inter], B [hidden, rank]

delta 以 float 叠加在 fake-quant 有效权重之后（output-space，见 ste.effective_weight），
B 初始化 0（训练起点 delta=0，与部署产物严格等价）。
"""
from __future__ import annotations

import math

import torch
from torch import nn

PROJ_SHAPES = {
    # proj 名 -> (out_mult, in_key)；gate_up 融合层 out = 2*inter
    "gate_up": ("gate_up", 2.0),
    "down": ("down", 1.0),
}


class ExpertLoRA(nn.Module):
    """per-expert LoRA 参数管理。

    Args:
        num_experts: 专家数
        hidden_size: 残差流维度
        moe_intermediate_size: 专家 intermediate 维度
        rank: LoRA 秩（Phase 1 建议 64）
        alpha: 缩放（effective delta = (B@A) * alpha/rank）
    """

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate_size: int,
                 rank: int = 64, alpha: float = 64.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.inter_size = moe_intermediate_size
        shape_a = {"gate_up": (rank, hidden_size), "down": (rank, moe_intermediate_size)}
        shape_b = {"gate_up": (2 * moe_intermediate_size, rank), "down": (hidden_size, rank)}
        self.params = nn.ParameterDict()
        for name, (out_dim, in_dim) in {
            "Agu": shape_a["gate_up"], "Bgu": shape_b["gate_up"],
            "Adn": shape_a["down"], "Bdn": shape_b["down"],
        }.items():
            self.params[name] = nn.Parameter(
                torch.zeros(num_experts, out_dim, in_dim, dtype=dtype)
            )
        for e in range(num_experts):  # A 侧 kaiming 初始化（B 保持 0）
            for a_name, fan_in in (("Agu", hidden_size), ("Adn", moe_intermediate_size)):
                nn.init.kaiming_uniform_(self.params[a_name].data[e], a=math.sqrt(5))

    def delta(self, proj: str, expert: int) -> torch.Tensor:
        """返回 [out, in] 的 LoRA 增量矩阵（gate_up: [2*inter, hidden]；down: [hidden, inter]）。"""
        if proj == "gate_up":
            a, b = self.params["Agu"][expert], self.params["Bgu"][expert]
        elif proj == "down":
            a, b = self.params["Adn"][expert], self.params["Bdn"][expert]
        else:
            raise KeyError(proj)
        return (b @ a) * self.scaling

    def trainable_parameters(self):
        return [p for p in self.params.values() if p.requires_grad]
