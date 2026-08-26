import json
import numpy as np

from actmask.experiments.milestone3r_nl_full import FAIR_KEYS, _metric


def test_full_evaluator_uses_only_preregistered_fair_input_keys():
    assert set(FAIR_KEYS) == {"history", "timestamps", "visibility", "observation_confidence", "candidate_actions", "nominal_action_timing", "tcp_state"}


def test_candidate_prefix_metric_keeps_complete_matched_pairs():
    rows = []
    for candidate in range(2):
        for branch in range(2):
            rows.append({"candidate_id": candidate, "signed_group": f"p{candidate}", "family": "f", "mechanism": "m", "world_id": 0, "branch": branch})
    labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
    scores = np.asarray([2.0, 0.0, 2.0, 0.0])
    metric = _metric(rows, labels, scores, np.arange(4), candidates=1)
    assert metric["examples"] == 2
    assert metric["pair_order_accuracy"] == 1.0
