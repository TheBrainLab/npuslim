"""Candidate-level diagnostics used by overlap-aware OPD routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


_EPS = 1e-12


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Teacher/student relation over the same candidate set."""

    top1_agree: bool
    topk_overlap: float
    student_entropy: float
    teacher_entropy: float
    entropy_gap: float
    student_margin: float
    teacher_margin: float
    kl_teacher_student: float
    symmetric_kl: float


def _as_float_list(values: Iterable[float]) -> list[float]:
    out = [float(v) for v in values]
    if len(out) < 2:
        raise ValueError("candidate diagnostics require at least two scores")
    return out


def stable_softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over candidate scores."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    xs = [float(x) / temperature for x in scores]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    denom = sum(exps) or 1.0
    return [x / denom for x in exps]


def entropy(probs: Sequence[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), _EPS)) for p in probs)


def topk_indices(scores: Sequence[float], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    return sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: min(k, len(scores))]


def argmax(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: float(scores[i]))


def margin(scores: Sequence[float]) -> float:
    order = topk_indices(scores, 2)
    return float(scores[order[0]]) - float(scores[order[1]]) if len(order) == 2 else 0.0


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("probability vectors must have the same length")
    return sum(
        float(pi) * (math.log(max(float(pi), _EPS)) - math.log(max(float(qi), _EPS)))
        for pi, qi in zip(p, q)
    )


def compute_diagnostics(
    student_scores: Iterable[float],
    teacher_scores: Iterable[float],
    *,
    topk: int = 2,
    temperature: float = 1.0,
) -> CandidateDiagnostics:
    """Compute overlap and entropy signals for one OPD candidate state."""

    student = _as_float_list(student_scores)
    teacher = _as_float_list(teacher_scores)
    if len(student) != len(teacher):
        raise ValueError("student and teacher scores must have the same number of candidates")

    k = min(topk, len(student))
    student_probs = stable_softmax(student, temperature=temperature)
    teacher_probs = stable_softmax(teacher, temperature=temperature)
    student_top = set(topk_indices(student, k))
    teacher_top = set(topk_indices(teacher, k))
    student_entropy = entropy(student_probs)
    teacher_entropy = entropy(teacher_probs)
    kl_ts = kl_divergence(teacher_probs, student_probs)
    kl_st = kl_divergence(student_probs, teacher_probs)

    return CandidateDiagnostics(
        top1_agree=argmax(student) == argmax(teacher),
        topk_overlap=len(student_top & teacher_top) / max(k, 1),
        student_entropy=student_entropy,
        teacher_entropy=teacher_entropy,
        entropy_gap=abs(teacher_entropy - student_entropy),
        student_margin=margin(student),
        teacher_margin=margin(teacher),
        kl_teacher_student=kl_ts,
        symmetric_kl=0.5 * (kl_ts + kl_st),
    )
