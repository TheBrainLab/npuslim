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


def smooth_scale_ab(x_scale, w_scale, alpha=0.5, beta=0.5, eps=1e-8):
    """FlexSmooth per-input-channel scale with independently searched beta.

    s = |X|^alpha / |W|^beta   (= msmodelslim flex_smooth FlexSmoothScaleCalculator)

    Unlike ``smooth_scale`` (which fixes beta = 1 - alpha), ``beta`` here is the
    value found by the alpha/beta grid search (``alpha_beta_search``), matching
    msmodelslim's flex_smooth_quant (best_practice.yaml leaves alpha/beta None ->
    the processor searches them per subgraph).
    """
    x_scale = x_scale.float().abs().clamp_min(eps)
    w_scale = w_scale.float().abs().clamp_min(eps)
    s = x_scale.pow(float(alpha)) / w_scale.pow(float(beta))
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
def awq_search(w, x_scale, nbits=4, grid=20, eps=1e-8, x_search=None):
    """Grid search for optimal AWQ per-input-channel scale s = x_scale^alpha.

    Matches modelslim flex_awq_ssz: s is a balanced scale (consumer x s, producer
    x 1/s) applied WITHOUT geometric normalization. The objective approximates the
    activation-weighted output quantization error of the scaled consumer weight:
    the producer's x1/s fold makes the effective activation (x_scale/s), so each
    element error is weighted by (x_scale/s)^2.

    modelslim searches the exponent with max|act|^alpha and applies the final
    scale with mean|act|^alpha (flex_awq_ssz.transform). Pass the abs-max
    statistic via x_search to reproduce that search; the returned scale is
    always built from x_scale (the mean statistic).

    Args:
        w: [out, in] float weight matrix (the consumer, e.g. down_proj)
        x_scale: [in] per-input-channel activation significance (abs-mean) —
            statistic of the RETURNED scale
        nbits: weight bit width (4 or 8)
        grid: number of alpha grid points
        eps: numerical stability
        x_search: [in] optional activation significance used ONLY for the alpha
            search objective (modelslim: abs-max). None → search with x_scale
            (legacy mean/mean behavior).

    Returns:
        best_s: [in] per-input-channel scale (x_scale^alpha_best, no normalization)
    """
    x_scale = x_scale.float().clamp_min(eps)
    xs = x_scale if x_search is None else x_search.float().clamp_min(eps)
    w = w.float()
    best_loss = float('inf')
    best_alpha = None

    rtn_fn = _rtn_sym_int4 if nbits == 4 else _rtn_sym_int8

    for i in range(grid):
        alpha = i / max(grid - 1, 1)
        s = xs.pow(alpha).clamp(min=eps)
        scaled_w = w * s.view(1, -1)
        delta = rtn_fn(scaled_w, perchannel=True) - scaled_w
        weight = (xs.view(1, -1) / s.view(1, -1)).pow(2)
        loss = (delta.pow(2) * weight).mean().item()

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    if best_alpha is None:
        return torch.ones_like(x_scale)
    return x_scale.pow(best_alpha).clamp(min=eps)


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
def awq_apply_scaled(w, s):
    """Apply AWQ balanced scaling (consumer side): w_scaled = w * s.

    Matches modelslim UpDownSubgraphFusion: the consumer (down_proj) keeps the
    scaling (NOT pre-quantized, NOT divided back), so the normal W4 quantization
    acts on the scaled weight and the producer's x(1/s) fold cancels it exactly.

    Args:
        w: [out, in] float weight (the consumer, e.g. down_proj)
        s: [in] per-input-channel scale

    Returns:
        w_scaled: [out, in] float weight (same dtype as input)
    """
    w = w.float()
    return (w * s.view(1, -1)).to(w.dtype)


@torch.no_grad()
def compute_weight_scale(w):
    """Per-input-channel weight abs-max for SmoothQuant.

    Args:
        w: [out, in] float weight
    Returns:
        w_scale: [in] per-input-channel abs-max
    """
    return w.float().abs().amax(dim=0)