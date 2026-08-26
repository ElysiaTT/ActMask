from actmask.experiments.milestone3r_nonlinear_ood import HELD_OUT


def test_ood_has_two_distinct_required_mechanisms():
    assert set(HELD_OUT)=={'jerk','regime_switch'}
