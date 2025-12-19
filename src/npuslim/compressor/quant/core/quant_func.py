
import torch


@torch.no_grad()
def quantize_weight_int(
    x: torch.Tensor, scales: torch.Tensor, bits=8
) -> tuple[torch.Tensor, float]:
    if scales.ndim == 2:  # weight group-wise
        scales = torch.repeat_interleave(scales, x.shape[1] // scales.shape[1], dim=-1)
    bnt = (1 << (bits - 1)) - 1

    while scales.ndim < x.ndim:
        scales = scales.unsqueeze(-1)
    scales.div_(bnt)
    x.div_(scales).round_().clamp_(-bnt - 1, bnt)
    return x, scales


if __name__ == "__main__":
    x = torch.randn(2,2).npu()
    scales = torch.randn(2,2).npu()
    print(scales.ndim)
    bits = 4
    print(x, scales)
    x, scales = quantize_weight_int(x, scales, bits)
    
    print(x, scales)
