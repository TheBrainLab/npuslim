"""Small candidate-score calibrator used by logit-level OPD."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - depends on optional torch install.
    torch = None
    nn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if nn is not None:

    class CandidateCalibrator(nn.Module):
        """Trainable score-level OPD adapter for choice-style tasks."""

        def __init__(self, max_choices: int) -> None:
            super().__init__()
            self.raw_scale = nn.Parameter(torch.tensor(1.0))
            self.norm_scale = nn.Parameter(torch.tensor(0.0))
            self.length_scale = nn.Parameter(torch.tensor(0.0))
            self.choice_bias = nn.Parameter(torch.zeros(max_choices))
            self.global_bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, scores: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            n = scores.numel()
            norm_scores = scores / lengths.clamp_min(1.0)
            centered_lengths = lengths - lengths.mean()
            return (
                self.raw_scale * scores
                + self.norm_scale * norm_scores
                + self.length_scale * centered_lengths
                + self.choice_bias[:n]
                + self.global_bias
            )

else:

    class CandidateCalibrator:  # type: ignore[no-redef]
        """Placeholder that reports a clear error when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("CandidateCalibrator requires torch") from _IMPORT_ERROR


__all__ = ["CandidateCalibrator"]
