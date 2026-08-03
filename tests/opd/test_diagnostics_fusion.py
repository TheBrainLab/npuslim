from npuslim.opd import ExpertScoreDelta, OverlapAwareRouter, compute_diagnostics, fuse_candidate_scores


def test_candidate_diagnostics_overlap_and_entropy():
    diagnostics = compute_diagnostics([1.0, 0.2, -0.5], [0.8, 0.5, -0.3], topk=2)

    assert diagnostics.top1_agree is True
    assert diagnostics.topk_overlap == 1.0
    assert diagnostics.entropy_gap >= 0.0
    assert diagnostics.kl_teacher_student >= 0.0


def test_fuse_candidate_scores_uses_continuous_expert_weight():
    router = OverlapAwareRouter(min_factor=0.0)
    fused, weights, diagnostics = fuse_candidate_scores(
        [1.0, 0.0],
        [0.8, 0.2],
        [ExpertScoreDelta(name="mc", scores=[0.6, 0.4], base_weight=0.5)],
        router=router,
    )

    assert diagnostics.top1_agree is True
    assert 0.0 < weights["mc"] <= 0.5
    assert fused[0] < 1.0
    assert fused[1] > 0.0
