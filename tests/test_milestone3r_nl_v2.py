import json
from pathlib import Path
import numpy as np

from actmask.experiments.milestone3r_nl_v2 import CONFIG_HASH, config_hash, swap_history
from actmask.experiments.milestone3r_nl_v2_probe import diversity, ood_overlap_audit


ROOT = Path("outputs/actmask/milestone3r_nl_v2")


def test_v2_preregistered_hash_is_stable_and_v1_remains_frozen():
    assert config_hash() == CONFIG_HASH
    v1 = json.loads(Path("outputs/actmask/milestone3r_nl_v1/preregistered_config.json").read_text())
    assert v1["extension"] == "milestone3r_nl_v1"


def test_pair_swap_reverses_only_early_frames_and_preserves_final_frames():
    h = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    changed = swap_history(h)
    assert np.array_equal(changed[:, :4], h[:, :4][:, ::-1])
    assert np.array_equal(changed[:, 4:], h[:, 4:])


def test_generated_v2_pairs_have_exact_action_and_swap_matching():
    report = json.loads((ROOT / "probe_final/id/generation_report.json").read_text())
    assert report["audit"]["max_matching_error"] <= 1e-6
    assert report["audit"]["pair_swap_max_error"] <= 1e-6
    assert report["audit"]["label_flip_rate"] == 1.0


def test_candidate_diversity_and_candidate_index_audit_pass():
    audit = diversity(ROOT / "probe_final/id")["summary"]
    assert audit["nondegenerate_world_fraction"] >= .8
    assert audit["mean_branch_outcome_change_rate"] >= .25
    assert audit["candidate_template_rate_span"] == 0.0


def test_ood_ranges_are_disjoint_and_mechanism_is_held():
    audit = ood_overlap_audit(ROOT / "probe_final")
    assert not audit["physical_parameter"]["damping_overlap"]
    assert not audit["physical_parameter"]["frequency_overlap"]
    assert not audit["temporal_delay"]["overlap"]
    assert not audit["held_mechanism"]["overlap"]


def test_probe_controls_and_ranking_are_non_degenerate():
    report = json.loads((ROOT / "evaluation/probe_evaluation.json").read_text())
    assert report["swap"]["counterfactual_swap_accuracy"] >= .8
    assert report["swap"]["score_swap_consistency"] > report["swap_controls"]["ActionMLP"]["score_swap_consistency"]
    assert report["ranking"]["dynamic"]["top1_success"] - report["ranking"]["ActionMLP"]["top1_success"] >= .1
