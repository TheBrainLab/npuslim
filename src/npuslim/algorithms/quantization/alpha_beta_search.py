#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FlexSmooth alpha/beta 搜索（移植自 msmodelslim flex_smooth/alpha_beta_search.py）。

语义（与参考逐行对齐）：
  s = act_max^alpha * w_max^(-beta)          # 即 act^α / w^β（per-input-channel）
  x' = x / s ; W' = W * s（consumer 输入列 ×s、producer 输出行 ×1/s 的等价 fold）
  MSE = ||int8(x')·int8(W')^T - fp(x,W)||_F / ||fp(x,W)||_F
搜索：phase1 网格搜 alpha（此时 beta=1-alpha）；phase2 固定 alpha 再网格搜 beta。

与参考的差异（性能优化，不改语义）：
  - fp_golden 只算一次（参考每个候选重算）；int8 模拟与参考一致。
  - act 用调用方给的采样 token（参考用全量激活，本端为控制耗时用子采样）。
"""
from typing import Tuple

import numpy as np
import torch


@torch.no_grad()
def quant_int8sym(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """对称 int8 模拟量化（per-last-dim max）。"""
    xmax = torch.abs(x).max(dim=dim, keepdim=True)[0]
    interval = (xmax / 127.0).clamp_min(1e-12)
    quanted = torch.round(x / interval).clip(-127, 127)
    return quanted * interval


@torch.no_grad()
def quant_int8asym(x: torch.Tensor) -> torch.Tensor:
    """非对称 int8 模拟量化（per-tensor）。"""
    x_max = torch.max(x)
    x_min = torch.min(x)
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(x_min)
    scale = torch.max((x_max - x_min) / 255, eps)
    zero_point = (-1 * x_min / scale).round() - 128
    qx = (x / scale + zero_point).round()
    return (qx - zero_point) * scale


class FlexSmoothAlphaBetaSearcher:
    """(alpha, beta) 网格搜索，最小化模拟量化后的归一化输出 MSE。"""

    def __init__(self, act_sym: bool = True, search_step: float = 0.05):
        self.act_sym = act_sym
        self.search_step = search_step

    @torch.no_grad()
    def evaluate_alpha_beta(self, act: torch.Tensor, weights: torch.Tensor,
                            alpha: float, beta: float,
                            golden: torch.Tensor | None = None) -> float:
        fp_golden = golden if golden is not None else torch.matmul(act, weights.T)
        normal = torch.mean(fp_golden ** 2) ** 0.5 + 1e-12
        act_max = torch.max(torch.abs(act), dim=0, keepdim=True)[0]
        w_max = torch.max(torch.abs(weights), dim=0, keepdim=True)[0]
        scale = act_max.pow(alpha) * w_max.pow(-beta)
        scale = scale.clamp_min(1e-12)
        scaled_act = act / scale
        scaled_weights = weights * scale
        quant_weight = quant_int8sym(scaled_weights)
        quant_act = (quant_int8sym(scaled_act) if self.act_sym
                     else quant_int8asym(scaled_act))
        quant_result = torch.matmul(quant_act, quant_weight.T)
        return (torch.mean((quant_result - fp_golden) ** 2) ** 0.5 / normal).item()

    @torch.no_grad()
    def search_alpha(self, act, weights, golden=None) -> Tuple[float, float]:
        best_alpha, best_mse = 0.0, float("inf")
        for alpha in np.round(np.arange(0.0, 1.0 + self.search_step, self.search_step), 2):
            mse = self.evaluate_alpha_beta(act, weights, float(alpha), 1.0 - float(alpha), golden)
            if mse <= best_mse:
                best_mse, best_alpha = mse, float(alpha)
        return best_alpha, best_mse

    @torch.no_grad()
    def search_beta(self, act, weights, alpha: float, best_mse_init: float,
                    golden=None) -> Tuple[float, float]:
        best_beta, best_mse = 0.0, best_mse_init
        for beta in np.round(np.arange(0.0, 1.0 + self.search_step, self.search_step), 2):
            mse = self.evaluate_alpha_beta(act, weights, alpha, float(beta), golden)
            if mse <= best_mse:
                best_mse, best_beta = mse, float(beta)
        return best_beta, best_mse

    @torch.no_grad()
    def search_alpha_beta(self, act: torch.Tensor, weights: torch.Tensor) -> Tuple[float, float, float]:
        act = act.float()
        weights = weights.float()
        golden = torch.matmul(act, weights.T)
        best_alpha, mse_alpha = self.search_alpha(act, weights, golden)
        best_beta, final_mse = self.search_beta(act, weights, best_alpha, mse_alpha, golden)
        return best_alpha, best_beta, final_mse
