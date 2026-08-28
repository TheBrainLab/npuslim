"""Scale search preprocessing for quantization (SmoothQuant / AWQ).

SmoothQuant: s = |X|^α / |W|^(1-α)  (per-input-channel, requires norm fusion)
AWQ:        grid search for optimal per-input-channel scaling s = |X|^α,
            then fold into weight: w_adjusted = RTN(w*s)/s

For MoE experts (no norm fusion), AWQ is used: the scale s is folded into
the weight values, producing a float weight with improved quantization
properties. GPTQ then quantizes this adjusted weight normally.

Reference: modelslim msmodelslim/processor/anti_outlier/awq/best_scales_search.py
           InstinctRazor src/quant/moe_ptq.py::awq_quant
"""

import torch


@torch.no_grad()
def smooth_scale(x_scale, w_scale, alpha=0.5, eps=1e-8):
    """SmoothQuant per-input-channel scale.

    s = |X|^α / |W|^(1-α)

    Args:
        x_scale: [in] per-input-channel activation abs-max
        w_scale: [in] per-input-channel weight abs-max
        alpha: migration strength (0=all to weight, 1=all to activation)
        eps: numerical stability

    Returns:
        s: [in] per-input-channel scale
    """
    x_scale = x_scale.float().clamp_min(eps)
    w_scale = w_scale.float().clamp_min(eps)
    s = (x_scale.pow(alpha) / w_scale.pow(1 - alpha))
    return s.clamp_min(eps)


@torch.no_grad()
def _rtn_sym_int4(w, perchannel=True):
    """Symmetric per-channel RTN int4 quantization. Returns float dequantized weights.

    Args:
        w: [out, in] float weight
        perchannel: if True, per-output-channel; else per-tensor

    Returns:
        w_q: [out, in] float dequantized weights
    """
    maxq = 7  # 2^(4-1) - 1 for symmetric int4
    if perchannel:
        scale = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / maxq
    else:
        scale = w.float().abs().max().clamp(min=1e-8) / maxq
    q = torch.round(w.float() / scale).clamp(-maxq - 1, maxq)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def _rtn_sym_int8(w, perchannel=True):
    """Symmetric per-channel RTN int8 quantization. Returns float dequantized weights."""
    maxq = 127  # 2^(8-1) - 1
    if perchannel:
        scale = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / maxq
    else:
        scale = w.float().abs().max().clamp(min=1e-8) / maxq
    q = torch.round(w.float() / scale).clamp(-maxq - 1, maxq)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def awq_search(w, x_scale, nbits=4, grid=20, eps=1e-8):
    """Grid search for optimal AWQ per-input-channel scale.

    For each alpha in [0, 1], s = (x_scale^α) / geometric_mean(x_scale^α),
    evaluates RTN(w*s)/s weight MSE, returns the best s.

    Args:
        w: [out, in] float weight matrix
        x_scale: [in] per-input-channel activation significance (abs-mean)
        nbits: weight bit width (4 or 8)
        grid: number of alpha grid points
        eps: numerical stability

    Returns:
        best_s: [in] per-input-channel scale
    """
    x_scale = x_scale.float().clamp_min(eps)
    w = w.float()
    best_loss = float('inf')
    best_s = None

    rtn_fn = _rtn_sym_int4 if nbits == 4 else _rtn_sym_int8

    for i in range(grid):
        alpha = i / max(grid - 1, 1)
        s = x_scale.pow(alpha).clamp(min=eps)
        s = s / (s.max() * s.min()).sqrt()  # geometric normalization (modelslim convention)

        w_adjusted = rtn_fn(w * s.view(1, -1), perchannel=True) / s.view(1, -1)
        loss = (w_adjusted - w).pow(2).mean().item()

        if loss < best_loss:
            best_loss = loss
            best_s = s.clone()

    if best_s is None:
        return torch.ones_like(x_scale)
    return best_s


@torch.no_grad()
def awq_apply(w, s, nbits=4):
    """Apply AWQ scaling: w_adjusted = RTN(w*s)/s.

    Returns a float weight with improved quantization properties.
    GPTQ will quantize this adjusted weight normally.

    Args:
        w: [out, in] float weight
        s: [in] per-input-channel scale
        nbits: weight bit width

    Returns:
        w_adjusted: [out, in] float weight (same dtype as input)
    """
    rtn_fn = _rtn_sym_int4 if nbits == 4 else _rtn_sym_int8
    w = w.float()
    w_adjusted = rtn_fn(w * s.view(1, -1), perchannel=True) / s.view(1, -1)
    return w_adjusted.to(w.dtype)


@torch.no_grad()
def compute_weight_scale(w):
    """Per-input-channel weight abs-max for SmoothQuant.

    Args:
        w: [out, in] float weight
    Returns:
        w_scale: [in] per-input-channel abs-max
    """
    return w.float().abs().amax(dim=0)