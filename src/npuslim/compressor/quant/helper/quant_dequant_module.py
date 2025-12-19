import torch
from torch import nn
from ..core.quant_func import quantize_weight_int


class INTDynQDQModule(nn.Module):
    def __init__(
        self,
        w_bits: int,
        weight: nn.Parameter,
        weight_scale: nn.Parameter,
        bias: nn.Parameter,
    ):
        super().__init__()

        quant_weight, weight_scale = quantize_weight_int(
            weight, weight_scale, bits=w_bits
        )
        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        output = torch.nn.functional.linear(
            x, self.weight * self.weight_scale, bias=self.bias
        )
        return output
