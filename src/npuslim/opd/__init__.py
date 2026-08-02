"""Overlap-aware on-policy distillation utilities.

The subpackage keeps OPD logic independent from quantization backends. Model
loading, W4A16 execution, and vLLM-Ascend integration stay in the existing
NPUSlim modules; OPD code consumes candidate scores/logits and returns adjusted
scores or routing weights.
"""

from npuslim.opd.diagnostics import CandidateDiagnostics, compute_diagnostics
from npuslim.opd.fusion import (
    ExpertScoreDelta,
    OverlapAwareRouter,
    fuse_candidate_scores,
)

__all__ = [
    "CandidateDiagnostics",
    "ExpertScoreDelta",
    "OverlapAwareRouter",
    "compute_diagnostics",
    "fuse_candidate_scores",
]
