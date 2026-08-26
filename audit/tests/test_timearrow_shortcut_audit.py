from __future__ import annotations

import gzip
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(AUDIT_ROOT))

import timearrow_shortcut_audit as audit  # noqa: E402


class PureAuditTests(unittest.TestCase):
    def test_arrow_action_rule_uses_payload_and_goal_positions(self) -> None:
        history = np.zeros((4, 6, 20), dtype=np.float32)
        actions = np.zeros((4, 20, 3), dtype=np.float32)
        history[0, 3, 0] = 1.0
        history[1, 3, 0] = -1.0
        history[2, 3, 17] = 1.0
        history[3, 3, 17] = -1.0
        actions[[0, 2], :, 0] = 1.0
        actions[[1, 3], :, 0] = -1.0
        arrow, action_sum, score = audit.arrow_action_score(history, actions)
        self.assertTrue(np.all(score > 0))
        axis, branch_hat, direction_hat = audit.observable_bits(arrow, action_sum)
        np.testing.assert_array_equal(axis, np.zeros(4, dtype=np.int64))
        np.testing.assert_array_equal(branch_hat, direction_hat)
        baseline = audit.ArrowActionCompatibility()
        self.assertEqual(baseline.name, "ArrowActionCompatibility")
        self.assertFalse(baseline.requires_training)
        np.testing.assert_allclose(baseline.score(history, actions), score)
        np.testing.assert_array_equal(baseline.predict(history, actions), score > 0)
        np.testing.assert_array_equal(baseline(history, actions), score > 0)

    def test_pair_order_tie_is_half(self) -> None:
        labels = np.asarray([True, False, False, True])
        scores = np.zeros(4)
        pairs = ["a", "a", "b", "b"]
        self.assertEqual(audit.pair_order_accuracy(labels, scores, pairs), 0.5)

    def test_rank_metrics_perfect_order(self) -> None:
        rows = []
        labels = []
        scores = []
        for branch in (0, 1):
            for candidate in range(5):
                label = candidate % 2 == branch
                rows.append(
                    {
                        "family": "f",
                        "mechanism": "m",
                        "world_id": 0,
                        "branch": branch,
                        "candidate_id": candidate,
                    }
                )
                labels.append(label)
                scores.append(1.0 if label else -1.0)
        result = audit.rank_by_history(np.asarray(labels), np.asarray(scores), rows, 5)
        self.assertEqual(result["contexts"], 2)
        self.assertEqual(result["top1_success"], 1.0)
        self.assertEqual(result["top3_success_recall"], 1.0)
        self.assertEqual(result["corrected_utility_gain_ndcg"], 1.0)
        self.assertEqual(result["normalized_regret"], 0.0)

    def test_exact_early_reverse_negates_symmetric_arrow(self) -> None:
        history = np.zeros((1, 6, 20), dtype=np.float32)
        history[0, :4, 0] = [-1.0, -0.3, 0.3, 1.0]
        actions = np.ones((1, 20, 3), dtype=np.float32)
        _, _, original = audit.arrow_action_score(history, actions)
        transformed = np.concatenate((history[:, :4][:, ::-1], history[:, 4:]), axis=1)
        _, _, reversed_score = audit.arrow_action_score(transformed, actions)
        np.testing.assert_allclose(reversed_score, -original)

    def test_multi_frame_linear_velocity_matches_known_slope(self) -> None:
        timestamps = np.asarray([[0.0, 0.2, 0.7, 1.5]], dtype=np.float32)
        history = np.zeros((1, 4, 20), dtype=np.float32)
        history[0, :, 0] = 2.5 * timestamps[0] + 4.0
        history[0, :, 17] = -1.25 * timestamps[0] - 2.0
        visibility = np.ones((1, 4, 1), dtype=np.float32)
        velocity = audit.multi_frame_linear_velocity(history, timestamps, visibility)
        self.assertAlmostEqual(float(velocity[0, 0]), 2.5, places=5)
        self.assertAlmostEqual(float(velocity[0, 17]), -1.25, places=5)

    def test_optional_evidence_validators_accept_success_and_blocked_records(self) -> None:
        environment = {
            "status": "passed",
            "requested_versions": {"python": "3.11"},
            "observed_versions": {"python": "3.11.15"},
            "smoke_import": True,
            "smoke_gpu": True,
            "pip_check": True,
            "core_environment_smoke_passed": True,
            "legacy_exact_reproduction": False,
            "differences": ["GPU model differs"],
            "commands": ["python smoke.py"],
            "logs": {"smoke": "environment/logs/smoke.log"},
        }
        self.assertIs(audit.validate_environment_evidence(environment), environment)
        blocked_environment = {
            "status": "blocked",
            "blocker": "Package mirror repeatedly truncated the wheel.",
            "requested_versions": {"torch": "2.6.0+cu124"},
            "observed_versions": {},
            "differences": ["torch is not installed"],
            "commands": ["pip install torch==2.6.0"],
            "logs": {"pip": "environment/logs/pip.log"},
        }
        self.assertIs(
            audit.validate_environment_evidence(blocked_environment), blocked_environment
        )
        replay = {
            "status": "completed_verified_exact",
            "verification_passed": True,
            "command": "python bounded_replay.py",
            "examples": 4,
            "pairs": 2,
            "attempted": 4,
            "schema_matches": True,
            "success_consistency": True,
            "filter_consistency": True,
            "scope_limitations": ["bounded re-execution, not snapshot replay"],
        }
        self.assertIs(audit.validate_replay_evidence(replay), replay)
        mismatched_replay = dict(replay, status="completed_with_mismatch", verification_passed=False)
        self.assertIs(audit.validate_replay_evidence(mismatched_replay), mismatched_replay)
        blocked_replay = {
            "status": "not_run_environment_blocked",
            "blocker": "Environment import smoke did not pass.",
            "commands": [],
            "scope_limitations": ["No simulator execution was attempted."],
        }
        self.assertIs(audit.validate_replay_evidence(blocked_replay), blocked_replay)

    def test_explicit_blocked_evidence_requires_actionable_blocker(self) -> None:
        with self.assertRaises(ValueError):
            audit.validate_replay_evidence(
                {"status": "blocked", "commands": [], "scope_limitations": []}
            )


class FrozenArtifactIntegrationTests(unittest.TestCase):
    @staticmethod
    def _gzip_data_rows(path: Path) -> int:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in handle) - 1

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_canonical_rule_and_parity(self) -> None:
        expected = {
            "id": 60000,
            "physical_parameter_ood": 8000,
            "temporal_delay_ood": 5600,
            "held_mechanism_ood": 12000,
        }
        for name, relative in audit.CANONICAL.items():
            with self.subTest(name=name):
                value = audit.dataset_summary(PROJECT_ROOT / relative, PROJECT_ROOT, canonical=True)
                self.assertEqual(value["examples"], expected[name])
                self.assertEqual(value["arrow_action_accuracy"], 1.0)
                self.assertEqual(value["metadata_parity_accuracy"], 1.0)
                self.assertEqual(value["zero_score_count"], 0)
                self.assertEqual(value["pair_integrity"]["label_flip_rate"], 1.0)

    def test_all_discovered_v2_datasets_share_observable_shortcut(self) -> None:
        paths = audit.discover_datasets(PROJECT_ROOT)
        self.assertEqual(len(paths), 30)
        summaries = [audit.dataset_summary(path, PROJECT_ROOT) for path in paths]
        self.assertEqual(sum(value["examples"] for value in summaries), 247344)
        self.assertTrue(all(value["arrow_action_accuracy"] == 1.0 for value in summaries))

    def test_canonical_attempted_and_filtered_counts(self) -> None:
        expected = {
            "id": (30000, 0),
            "physical_parameter_ood": (4000, 0),
            "temporal_delay_ood": (6000, 3200),
            "held_mechanism_ood": (6000, 0),
        }
        for name, relative in audit.CANONICAL.items():
            with self.subTest(name=name):
                value = audit.dataset_summary(PROJECT_ROOT / relative, PROJECT_ROOT)
                outcomes = value["outcomes"]
                self.assertEqual(outcomes["attempted_pairs"], expected[name][0])
                self.assertEqual(outcomes["nonflip_pairs"], expected[name][1])
                self.assertTrue(outcomes["retained_keys_equal_flip_keys"])
                self.assertTrue(outcomes["outcome_key_unique"])
                self.assertEqual(outcomes["outcome_duplicate_key_rows"], 0)
                self.assertTrue(outcomes["pair_flip_field_all_exact"])
                self.assertEqual(outcomes["pair_flip_field_mismatches"], 0)
                self.assertTrue(outcomes["metadata_pair_branch_unique"])
                self.assertEqual(outcomes["metadata_duplicate_pair_branch_rows"], 0)
                direct = outcomes["direct_join"]
                self.assertEqual(direct["joined_rows"], value["examples"])
                self.assertEqual(direct["ambiguous_or_missing_rows"], 0)
                self.assertEqual(direct["coverage"], 1.0)
                self.assertEqual(direct["metadata_success_match_rate"], 1.0)
                self.assertEqual(direct["label_success_match_rate"], 1.0)
                self.assertEqual(direct["metadata_label_match_rate"], 1.0)
                self.assertTrue(direct["all_exact"])

    def test_source_manifest_covers_every_discovered_dataset_input(self) -> None:
        datasets = audit.discover_datasets(PROJECT_ROOT)
        manifest = audit.source_manifest(PROJECT_ROOT, datasets)
        paths = [item["path"] for item in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        expected: set[str] = set()
        for root in datasets:
            for name in ("model_inputs.npz", "labels.npz", "metadata.jsonl"):
                expected.add(str((root / name).relative_to(PROJECT_ROOT)))
            outcomes = root / "candidate_outcomes.jsonl"
            if outcomes.exists():
                expected.add(str(outcomes.relative_to(PROJECT_ROOT)))
        self.assertEqual(len(expected), 105)
        self.assertEqual(manifest["discovered_dataset_directories"], 30)
        self.assertEqual(manifest["dataset_input_files"], len(expected))
        self.assertTrue(expected.issubset(set(paths)))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))
        for key_source in (
            "actmask/experiments/milestone3p_maniskill_smoke.py",
            "actmask/data/maniskill_pilot.py",
            "actmask/data/milestone3q_signed.py",
        ):
            self.assertIn(key_source, paths)
        audit_sources = {
            str(path.relative_to(PROJECT_ROOT))
            for path in AUDIT_ROOT.rglob("*.py")
            if "__pycache__" not in path.relative_to(AUDIT_ROOT).parts
        }
        self.assertTrue(audit_sources.issubset(set(paths)))
        saved = json.loads(
            (AUDIT_ROOT / "results/source_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved, manifest)

    def test_generated_csv_row_counts_and_optional_evidence_schema(self) -> None:
        results = AUDIT_ROOT / "results"
        expected_rows = {
            "canonical_predictions.csv.gz": 85600,
            "id_test_candidate_perturbations.csv.gz": 60000,
            "id_test_temporal_perturbations.csv.gz": 60000,
        }
        for name, expected in expected_rows.items():
            with self.subTest(name=name):
                path = results / name
                self.assertTrue(path.exists(), f"missing generated evidence: {path}")
                self.assertEqual(self._gzip_data_rows(path), expected)
        environment_path = AUDIT_ROOT / "environment/environment_status.json"
        if environment_path.exists():
            environment = audit.validate_environment_evidence(
                json.loads(environment_path.read_text(encoding="utf-8"))
            )
            if not audit._is_explicitly_blocked(environment):
                self.assertEqual(environment["status"], "passed")
                self.assertTrue(environment["core_environment_smoke_passed"])
        replay_path = AUDIT_ROOT / "replay/replay_verification.json"
        if replay_path.exists():
            replay = audit.validate_replay_evidence(
                json.loads(replay_path.read_text(encoding="utf-8"))
            )
            if not audit._is_explicitly_blocked(replay):
                self.assertEqual(
                    replay["verification_passed"],
                    replay["status"] == "completed_verified_exact",
                )

        manifest_path = results / "generated_manifest.json"
        self.assertTrue(manifest_path.exists())
        generated = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_paths = {item["path"] for item in generated["files"]}
        self.assertNotIn("results/generated_manifest.json", generated_paths)
        self.assertIn("audit_report.md", generated_paths)
        for directory in (AUDIT_ROOT / "environment", AUDIT_ROOT / "replay"):
            if directory.exists():
                expected = {
                    str(path.relative_to(AUDIT_ROOT))
                    for path in directory.rglob("*")
                    if path.is_file()
                    and "__pycache__" not in path.relative_to(AUDIT_ROOT).parts
                    and path.suffix not in {".pyc", ".pyo"}
                }
                self.assertTrue(expected.issubset(generated_paths))
        for item in generated["files"]:
            with self.subTest(generated_hash=item["path"]):
                path = AUDIT_ROOT / item["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, item["bytes"])
                self.assertEqual(self._sha256(path), item["sha256"])

    def test_generated_summary_declares_shortcut_saturation(self) -> None:
        path = AUDIT_ROOT / "results/summary.json"
        self.assertTrue(path.exists(), "run timearrow_shortcut_audit.py before artifact verification")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["benchmark_status"], "shortcut-saturated")
        self.assertEqual(
            summary["zero_training_baseline"],
            {"name": "ArrowActionCompatibility", "requires_training": False},
        )
        self.assertEqual(summary["canonical_totals"]["retained_examples"], 85600)
        self.assertEqual(summary["canonical_totals"]["retained_arrow_action_accuracy"], 1.0)
        direct = summary["canonical_totals"]["direct_outcome_join"]
        self.assertEqual(direct["joined_rows"], 85600)
        self.assertEqual(direct["coverage"], 1.0)
        self.assertEqual(direct["metadata_success_match_rate"], 1.0)
        self.assertEqual(direct["label_success_match_rate"], 1.0)
        self.assertTrue(direct["all_exact"])
        self.assertTrue(direct["all_outcome_keys_unique"])
        self.assertTrue(direct["all_metadata_pair_branch_keys_unique"])
        self.assertFalse(summary["direct_fabrication_evidence_found"])
        analytic = summary["analytic_coordinate_audit"]
        self.assertEqual(analytic["estimator"].split(" (")[0], "MultiFrameLinearVelocity")
        self.assertEqual(analytic["published_goal_only_test_pair_order"], 0.6)
        self.assertEqual(analytic["corrected_rule_test_pair_order"], 1.0)
        inventory = summary["dataset_inventory"]
        self.assertEqual(inventory["directories"], 30)
        self.assertEqual(inventory["all_directories_arrow_action_accuracy"], 1.0)

        candidate = summary["perturbations"]["candidate"]
        self.assertEqual(candidate["metadata_candidate_parity_remap_xor_one"]["accuracy"], 0.0)
        self.assertTrue(
            all(
                value["joint_record_order_shuffle"]["invariant"]
                for value in candidate[
                    "joint_candidate_record_order_shuffle_and_action_reassignment"
                ].values()
            )
        )
        temporal = summary["perturbations"]["temporal"]
        self.assertEqual(temporal["exact_early_partner_reverse"]["accuracy"], 0.0)
        self.assertEqual(
            temporal["exact_early_partner_reverse"]["accuracy_against_partner_labels"], 1.0
        )
        self.assertEqual(temporal["last_two_only"]["pair_order_accuracy"], 0.5)
        self.assertTrue(
            all(
                0.45 <= value["pair_order_accuracy"] <= 0.55
                for value in temporal["genuine_pair_shared_random_early_shuffle"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
