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


def compute_scales_with_zero(x, bits=8, sym=False, perchannel=True):
    maxq = torch.tensor(2**bits - 1)
    shape = x.shape

    if perchannel:
        x = x.flatten(1)
    else:
        x = x.flatten().unsqueeze(0)
    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if maxq < 0:
        scale = xmax
        zero = xmin
    else:
        scale = (xmax - xmin) / maxq
        if sym:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)
    shape = [-1] + [1] * (len(shape) - 1)
    scale = scale.reshape(shape)
    zero = zero.reshape(shape)
    return scale, zero


if __name__ == "__main__":
    x = torch.randn(2, 2).npu()
    scales = torch.randn(2, 2).npu()
    print(scales.ndim)
    bits = 4
    print(x, scales)
    x, scales = quantize_weight_int(x, scales, bits)

    print(x, scales)
