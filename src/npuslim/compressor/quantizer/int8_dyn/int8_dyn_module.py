import torch
from torch import nn
from ...core.quant_func import quantize_weight_int


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
        weight_scale = weight_scale.view(-1) if weight_scale.ndim == 0 else weight_scale
        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        output = torch.nn.functional.linear(
            x, self.weight * self.weight_scale, bias=self.bias
        )
        return output

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_rename = [k for k in state_dict.keys() if "weight_scale_int4" in k]
        for old_key in keys_to_rename:
            new_key = old_key.replace("weight_scale_int4", "weight_scale.int4")
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict
