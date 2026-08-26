import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_icra2027_real_robot_matched_reexecution.py"
PREREG = ROOT / "docs/icra2027_real_robot_matched_reexecution_prereg.json"


def load_module():
    spec = importlib.util.spec_from_file_location("real_robot_verifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prereg_hash():
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def compliant_data():
    candidate_time = [0.0, 0.5, 0.75]
    families = {
        "linear_x_intercept": [1.0, 0.0, 0.0],
        "linear_y_intercept": [0.0, 1.0, 0.0],
        "circular_intercept": [2**-0.5, 2**-0.5, 0.0],
    }
    base = [0.0, 0.0, 0.1]

    def point(direction, scalar):
        return [base[axis] + scalar * direction[axis] for axis in range(3)]

    trials = []
    for family, direction in families.items():
        candidate = [point(direction, scalar) for scalar in (0.0, 0.05, 0.1)]
        early_a = [point(direction, scalar) for scalar in (-0.03, -0.01, 0.01, 0.03)]
        early_b = list(reversed(early_a))
        for index in range(12):
            a_first = index % 2 == 0
            for branch in ("A", "B"):
                success = branch == "A"
                trials.append({
                    "family": family,
                    "pair_id": f"{family}-pair-{index:02d}",
                    "branch": branch,
                    "execution_order": 1 if (branch == "A") == a_first else 2,
                    "history_time_s": [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0],
                    "history_target_xyz_m": (early_a if branch == "A" else early_b)
                    + [base[:], base[:]],
                    "decision_target_xyz_m": base[:],
                    "decision_target_velocity_m_s": [0.0, 0.0, 0.0],
                    "decision_tcp_xyz_m": base[:],
                    "decision_joint_rad": [0.0] * 6,
                    "decision_joint_velocity_rad_s": [0.0] * 6,
                    "candidate_time_s": candidate_time[:],
                    "candidate_trajectory": [value[:] for value in candidate],
                    "candidate_space": "tcp_xyz_m",
                    "post_time_s": [0.0, 0.5, 0.75],
                    "post_target_xyz_m": (
                        [point(direction, scalar) for scalar in (0.0, 0.05, 0.12)]
                        if success
                        else [point(direction, 0.2) for _ in range(3)]
                    ),
                    "post_tcp_xyz_m": [value[:] for value in candidate],
                    "tracking_valid": [True, True, True],
                    "success": success,
                    "outcome_source": "measured_tracking",
                    "safety_abort": False,
                })
    return {
        "schema": "actmask-real-robot-matched-data-v1",
        "prereg_sha256": prereg_hash(),
        "metadata": {
            "hardware_id": "synthetic-test-arm",
            "fixture_id": "synthetic-test-fixture",
            "tracking_system": "synthetic-test-tracker",
            "calibration_sha256": "1" * 64,
            "candidate_controller_sha256": "2" * 64,
            "fixture_controller_sha256": "3" * 64,
            "collection_started_utc": "2026-08-09T00:00:00Z",
            "operator_safety_confirmation": True,
        },
        "trials": trials,
    }


def test_compliant_matched_cell_passes_all_gates():
    module = load_module()
    prereg = json.loads(PREREG.read_text())
    report = module.verify(compliant_data(), prereg, prereg_hash())
    assert report["passed"] is True
    assert report["counts"]["complete_pairs"] == 36
    assert report["counts"]["correct"] == 36
    assert report["counts"]["wrong"] == 0
    assert report["counts"]["tied"] == 0
    assert all(row["correct"] == 12 for row in report["counts"]["by_family"].values())
    assert report["physical_counterfactual_decision"] == "admit"
    assert report["learned_method_decision"] == "reject_analytic_saturation"
    assert report["ladder"]["action_only"]["pair_order"] == 0.5
    assert report["ladder"]["current_only"]["pair_order"] == 0.5
    assert report["ladder"]["unordered_history"]["pair_order"] == 0.5
    assert report["ladder"]["final_two_velocity"]["pair_order"] == 0.5
    assert report["ladder"]["multi_frame_linear_velocity"]["pair_order"] == 1.0
    assert report["ladder"]["early_endpoint_velocity"]["pair_order"] == 1.0
    assert report["provenance"]["prereg_sha256"] == prereg_hash()
    assert report["provenance"]["hardware_id"] == "synthetic-test-arm"
    assert len(report["provenance"]["candidate_sha256_by_family"]) == 3
    assert report["maximum_candidate_tracking_error_m"] == 0.0
    assert all(
        row["multi_frame_linear_velocity"]["pair_order"] == 1.0
        for row in report["ladder_by_family"].values()
    )
    assert all(row["status"] == "PASS" for row in report["gates"])


def test_candidate_change_is_rejected():
    module = load_module()
    prereg = json.loads(PREREG.read_text())
    data = compliant_data()
    data["trials"][1]["candidate_trajectory"][1][0] = 0.051
    report = module.verify(data, prereg, prereg_hash())
    assert report["passed"] is False
    gate = next(row for row in report["gates"] if row["name"] == "identical_candidate_action")
    assert gate["status"] == "FAIL"


def test_robot_feedback_that_does_not_follow_candidate_is_rejected():
    module = load_module()
    prereg = json.loads(PREREG.read_text())
    data = compliant_data()
    data["trials"][0]["post_tcp_xyz_m"][1][0] += 0.01
    report = module.verify(data, prereg, prereg_hash())
    assert report["passed"] is False
    gate = next(row for row in report["gates"] if row["name"] == "candidate_execution_fidelity")
    assert gate["status"] == "FAIL"
    assert gate["detail"]["maximum_error_m"] > gate["detail"]["tolerance_m"]


def test_aggregate_success_cannot_hide_a_missing_family():
    module = load_module()
    prereg = json.loads(PREREG.read_text())
    data = compliant_data()
    data["trials"] = [
        row for row in data["trials"] if row["family"] != "circular_intercept"
    ]
    report = module.verify(data, prereg, prereg_hash())
    assert report["passed"] is False
    gate = next(row for row in report["gates"] if row["name"] == "family_coverage")
    assert gate["status"] == "FAIL"


def test_aggregate_ordered_score_cannot_hide_one_failed_family():
    module = load_module()
    prereg = json.loads(PREREG.read_text())
    data = compliant_data()
    selected = {
        f"circular_intercept-pair-{index:02d}" for index in range(4)
    }
    pairs = {}
    for row in data["trials"]:
        if row["pair_id"] in selected:
            pairs.setdefault(row["pair_id"], {})[row["branch"]] = row
    for branches in pairs.values():
        branches["A"]["history_target_xyz_m"], branches["B"]["history_target_xyz_m"] = (
            branches["B"]["history_target_xyz_m"],
            branches["A"]["history_target_xyz_m"],
        )
    report = module.verify(data, prereg, prereg_hash())
    assert report["ladder"]["multi_frame_linear_velocity"]["pair_order"] >= 0.8
    assert (
        report["ladder_by_family"]["circular_intercept"]
        ["multi_frame_linear_velocity"]["pair_order"]
        < prereg["ladder"]["minimum_family_ordered_pair_order"]
    )
    gate = next(row for row in report["gates"] if row["name"] == "ordered_history_signal")
    assert gate["status"] == "FAIL"


def test_cli_writes_the_same_admission_report(tmp_path):
    data_path = tmp_path / "matched.json"
    output_path = tmp_path / "verification.json"
    data_path.write_text(json.dumps(compliant_data()))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout_report = json.loads(completed.stdout)
    stored_report = json.loads(output_path.read_text())
    assert stdout_report == stored_report
    assert stored_report["passed"] is True
    assert stored_report["learned_method_decision"] == "reject_analytic_saturation"
