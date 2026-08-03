"""Continuous expert fusion for overlap-aware OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from npuslim.opd.diagnostics import CandidateDiagnostics, compute_diagnostics


@dataclass(frozen=True)
class ExpertScoreDelta:
    """Candidate-score correction produced by one OPD expert."""

    name: str
    scores: Sequence[float]
    base_weight: float = 1.0
    stability: float = 1.0
    task_weight: float = 1.0


class OverlapAwareRouter:
    """Compute continuous OPD expert weights from candidate diagnostics."""

    def __init__(
        self,
        *,
        min_factor: float = 0.15,
        entropy_scale: float = 1.0,
        low_margin_threshold: float = 0.15,
        disagreement_penalty: float = 0.55,
    ) -> None:
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("min_factor must be in [0, 1]")
        self.min_factor = min_factor
        self.entropy_scale = entropy_scale
        self.low_margin_threshold = low_margin_threshold
        self.disagreement_penalty = disagreement_penalty

    def factor(self, diagnostics: CandidateDiagnostics) -> float:
        """Return a sample-level safety factor in [min_factor, 1]."""

        factor = 1.0
        if not diagnostics.top1_agree:
            factor *= self.disagreement_penalty
        factor *= 0.5 + 0.5 * diagnostics.topk_overlap
        factor *= 1.0 / (1.0 + self.entropy_scale * diagnostics.entropy_gap)
        if diagnostics.student_margin < self.low_margin_threshold:
            factor *= 0.75
        return max(self.min_factor, min(1.0, factor))

    def expert_weight(
        self,
        expert: ExpertScoreDelta,
        diagnostics: CandidateDiagnostics,
        *,
        task_overrides: Mapping[str, float] | None = None,
    ) -> float:
        """Return the final continuous weight for one expert."""

        override = 1.0
        if task_overrides is not None:
            override = float(task_overrides.get(expert.name, 1.0))
        return (
            float(expert.base_weight)
            * float(expert.task_weight)
            * float(expert.stability)
            * override
            * self.factor(diagnostics)
        )


def _as_scores(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def fuse_candidate_scores(
    student_scores: Iterable[float],
    teacher_scores: Iterable[float],
    experts: Sequence[ExpertScoreDelta],
    *,
    router: OverlapAwareRouter | None = None,
    task_overrides: Mapping[str, float] | None = None,
    topk: int = 2,
    temperature: float = 1.0,
) -> tuple[list[float], dict[str, float], CandidateDiagnostics]:
    """Fuse base student candidate scores with multiple OPD expert corrections.

    The returned scores implement:

        z_final = z_student + sum_i lambda_i * (z_expert_i - z_student)

    where lambda_i is a continuous, overlap-aware expert weight.
    """

    base = _as_scores(student_scores)
    diagnostics = compute_diagnostics(base, teacher_scores, topk=topk, temperature=temperature)
    router = router or OverlapAwareRouter()
    fused = list(base)
    weights: dict[str, float] = {}

    for expert in experts:
        scores = _as_scores(expert.scores)
        if len(scores) != len(base):
            raise ValueError(f"expert {expert.name!r} has {len(scores)} scores, expected {len(base)}")
        weight = router.expert_weight(expert, diagnostics, task_overrides=task_overrides)
        weights[expert.name] = weight
        for i, (student_score, expert_score) in enumerate(zip(base, scores)):
            fused[i] += weight * (expert_score - student_score)

    return fused, weights, diagnostics
