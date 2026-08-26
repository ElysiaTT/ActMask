import numpy as np

from actmask.experiments.milestone3r_nl_v2_rankfix import rank_by_history


def test_rank_by_history_does_not_pool_counterfactual_branches():
    rows = []
    for branch in (0, 1):
        for candidate in (0, 1):
            rows.append(
                {
                    "family": "task",
                    "mechanism": "mechanism",
                    "world_id": 7,
                    "branch": branch,
                    "candidate_id": candidate,
                }
            )
    labels = np.asarray([1, 0, 0, 1], dtype=np.float32)
    scores = np.asarray([0.9, 0.1, 0.8, 0.7], dtype=np.float32)
    result = rank_by_history(labels, scores, rows, np.arange(4), 2)

    assert result["contexts"] == 2
    assert result["feasible_contexts"] == 2
    assert result["mixed_contexts"] == 2
    assert result["top1_success"] == 0.5
    assert result["normalized_regret"] == 0.5


def test_rank_by_history_marks_an_infeasible_prefix():
    rows = []
    for branch in (0, 1):
        for candidate in (0, 1):
            rows.append(
                {
                    "family": "task",
                    "mechanism": "mechanism",
                    "world_id": 7,
                    "branch": branch,
                    "candidate_id": candidate,
                }
            )
    labels = np.asarray([1, 1, 0, 0], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    result = rank_by_history(labels, scores, rows, np.arange(4), 2)

    assert result["contexts"] == 2
    assert result["feasible_contexts"] == 1
    assert result["feasible_context_fraction"] == 0.5
    assert result["mixed_contexts"] == 0
    assert result["top1_success_when_feasible"] == 1.0
    assert result["normalized_regret"] == 0.0
