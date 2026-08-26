from __future__ import annotations

import inspect
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.audit import (
    binding_breaking_shuffle,
    centered_twin_preference,
    evaluate_all,
)
from actmask.experiments.history_action_binding_v2.common import PROTOCOL as V2_PROTOCOL
from actmask.experiments.history_action_binding_v2.symbolic import symbolic_dataset as v2_symbolic_dataset
from actmask.experiments.history_action_binding_v2_1.audit import METRIC_KEY, slot_balance_table
from actmask.experiments.history_action_binding_v2_1.common import OUTPUT_ROOT, PROJECT_ROOT, PROTOCOL, TASKS, scientific_core_manifest
from actmask.experiments.history_action_binding_v2_1.freeze import verify_protected_boundaries
from actmask.experiments.history_action_binding_v2_1.generate import symbolic_allows_gpu
from actmask.experiments.history_action_binding_v2_1.symbolic import candidate_sampler, symbolic_dataset


class ProtocolRepairTest(unittest.TestCase):
    def test_cyclic_latin_square_is_exact_for_any_consecutive_64_seeds(self):
        for base_seed in (0, 1, 15, 16, 917_351):
            for task in TASKS:
                data = symbolic_dataset(task, 64, base_seed, "test")
                table = slot_balance_table(data["candidate_template_id"])
                self.assertEqual(table.shape, (16, 16))
                self.assertTrue(np.array_equal(table, np.full((16, 16), 4)))
                for row in data["candidate_template_id"]:
                    self.assertEqual(set(row.tolist()), set(range(16)))

    def test_candidate_sampler_is_public_only_and_deterministic(self):
        self.assertEqual(
            tuple(inspect.signature(candidate_sampler).parameters),
            ("task", "public_rotation", "candidate_seed"),
        )
        source = inspect.getsource(candidate_sampler).lower()
        for forbidden in ("latent_phi", "alpha", "beta", "branch", "success", "utility", "label", "future"):
            self.assertNotIn(forbidden, source)
        one = candidate_sampler(TASKS[0], 0.37, 12345)
        two = candidate_sampler(TASKS[0], 0.37, 12345)
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(one, two)))

    def test_twin_marginals_current_and_candidates_are_exact(self):
        for task in TASKS:
            data = symbolic_dataset(task, 64, 78431, "test")
            self.assertTrue(np.array_equal(data["probe_actions"][:, 0], data["probe_actions"][:, 1]))
            self.assertTrue(np.array_equal(
                np.sort(data["probe_delta"][:, 0], axis=1),
                np.sort(data["probe_delta"][:, 1], axis=1),
            ))
            self.assertTrue(np.array_equal(
                np.sort(data["probe_utility"][:, 0], axis=1),
                np.sort(data["probe_utility"][:, 1], axis=1),
            ))
            self.assertTrue(np.array_equal(
                data["decision_public_state"][:, 0], data["decision_public_state"][:, 1]
            ))
            self.assertTrue(np.array_equal(
                data["candidate_actions_by_branch"][:, 0],
                data["candidate_actions_by_branch"][:, 1],
            ))

    def test_candidate_actions_are_novel(self):
        candidate_magnitudes = set(PROTOCOL["candidate_magnitudes"])
        for key in ("task_a_probe_magnitudes", "task_b_probe_magnitudes"):
            self.assertTrue(candidate_magnitudes.isdisjoint(PROTOCOL[key]))
        probe_angles = set(PROTOCOL["probe_directions_degrees"])
        self.assertTrue(probe_angles.isdisjoint(PROTOCOL["task_a_candidate_directions_degrees"]))
        task_b_angles = {
            (angle + PROTOCOL["task_b_candidate_direction_offset_degrees"]) % 360.0
            for angle in probe_angles
        }
        self.assertTrue(probe_angles.isdisjoint(task_b_angles))

    def test_corrected_metric_rejects_branch_constants(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(20, 2, 16))
        constants = np.broadcast_to(np.asarray([[-1.0], [2.0]])[None], truth.shape)
        templates = np.tile(np.arange(16), (20, 1))
        score, *_ = centered_twin_preference(truth, constants, templates)
        self.assertEqual(score, 0.5)

    def test_binding_break_preserves_marginals_but_changes_correspondence(self):
        data = symbolic_dataset(TASKS[0], 8, 9191, "test")
        actions, results = binding_breaking_shuffle(data["probe_actions"], data["probe_delta"])
        self.assertTrue(np.array_equal(actions, data["probe_actions"]))
        self.assertTrue(np.array_equal(np.sort(results, axis=2), np.sort(data["probe_delta"], axis=2)))
        self.assertFalse(np.array_equal(results, data["probe_delta"]))

    def test_task_a_repair_has_signal_without_interaction_shortcut(self):
        data = symbolic_dataset(TASKS[0], 64, PROTOCOL["development_seeds"][0], "test")
        _, metric_sets, _ = evaluate_all(data)
        original = metric_sets["original"]
        self.assertGreaterEqual(original["quadratic_fourier_sysid"][METRIC_KEY], 0.90)
        self.assertLess(original["nearest_historical_action"][METRIC_KEY], 0.90)
        self.assertLess(original["knn_action_result_lookup"][METRIC_KEY], 0.90)
        self.assertEqual(original["arrow_action_compatibility"][METRIC_KEY], 0.5)

    def test_task_b_scientific_protocol_unchanged_except_slots(self):
        for key in (
            "task_b_probe_magnitudes", "candidate_magnitudes", "probe_directions_degrees",
            "task_b_alpha_range", "task_b_beta_range", "task_b_hidden_phase_jitter_degrees",
            "twin_phase_offset_degrees", "symbolic_displacement_gain", "goal_x", "success_radius",
        ):
            self.assertEqual(PROTOCOL[key], V2_PROTOCOL[key])
        self.assertEqual(
            PROTOCOL["task_b_candidate_direction_offset_degrees"],
            V2_PROTOCOL["candidate_direction_offset_degrees"],
        )
        old = v2_symbolic_dataset(TASKS[1], 8, 45631)
        new = symbolic_dataset(TASKS[1], 8, 45631, "test")
        for key in ("latent_phi", "latent_alpha", "latent_beta", "probe_actions", "probe_delta"):
            self.assertTrue(np.array_equal(old[key], new[key]), key)

    def test_failed_symbolic_summary_never_allows_gpu(self):
        failure = {
            "passed": False,
            "tasks": {task: {"passed": task == TASKS[0]} for task in TASKS},
        }
        self.assertFalse(symbolic_allows_gpu(failure))
        success = {"passed": True, "tasks": {task: {"passed": True} for task in TASKS}}
        self.assertTrue(symbolic_allows_gpu(success))


class FrozenEvidenceTest(unittest.TestCase):
    def test_freeze_and_official_are_consistent_if_present(self):
        freeze_path = OUTPUT_ROOT / "freeze_record.json"
        if not freeze_path.exists():
            self.skipTest("not frozen yet")
        freeze = json.loads(freeze_path.read_text())
        self.assertEqual(freeze["status"], "FROZEN_BEFORE_OFFICIAL_PREFLIGHT")
        self.assertEqual(freeze["scientific_core_manifest"], scientific_core_manifest())
        started = OUTPUT_ROOT / "official_symbolic_preflight" / "official_run_started.json"
        if started.exists():
            record = json.loads(started.read_text())
            self.assertEqual(record["status"], "STARTED_ONCE")
            self.assertEqual(record["official_seed"], freeze["official_seed"])

    def test_protected_boundaries_if_frozen(self):
        if not (OUTPUT_ROOT / "freeze_record.json").exists():
            self.skipTest("not frozen yet")
        self.assertTrue(verify_protected_boundaries()["passed"])

    def test_official_artifacts_and_one_shot_marker_if_present(self):
        root = OUTPUT_ROOT / "official_symbolic_preflight"
        summary_path = root / "symbolic_preflight_summary.json"
        if not summary_path.exists():
            self.skipTest("official preflight not run yet")
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["official_run_count"], 1)
        self.assertTrue((root / "official_run_started.json").exists())
        self.assertTrue((root / "official_run_completed.json").exists())
        for task in TASKS:
            item = summary["tasks"][task]
            self.assertEqual(item["details"]["slot_min"], 4)
            self.assertEqual(item["details"]["slot_max"], 4)
            self.assertTrue((root / task / "symbolic_samples.npz").exists())
            self.assertTrue((root / task / "baseline_predictions.csv").exists())
            self.assertTrue((root / task / "intervention_samples.npz").exists())

    def test_symbolic_failure_is_fail_closed_if_terminal(self):
        path = OUTPUT_ROOT / "summary.json"
        if not path.exists():
            self.skipTest("not finalized")
        summary = json.loads(path.read_text())
        if summary["symbolic_preflight"]["passed"]:
            self.skipTest("official symbolic passed")
        self.assertEqual(summary["gate"], "SYMBOLIC_PREFLIGHT_FAIL")
        self.assertFalse((OUTPUT_ROOT / "smoke").exists())
        self.assertFalse((OUTPUT_ROOT / "pilot").exists())
        self.assertEqual(summary["candidate_executions"], 0)
        self.assertFalse(summary["learned_training_authorized"])

    def test_frozen_smoke_execution_error_stops_gpu_if_present(self):
        error_path = OUTPUT_ROOT / "smoke_execution_error.json"
        if not error_path.exists():
            self.skipTest("no frozen smoke execution error")
        record = json.loads(error_path.read_text())
        self.assertEqual(record["status"], "GPU_SMOKE_EXECUTION_ERROR")
        self.assertFalse(record["gpu_environment_created"])
        self.assertFalse(record["gym_make_reached"])
        self.assertEqual(record["candidate_executions"], 0)
        self.assertTrue(record["scientific_core_frozen_unchanged"])
        self.assertFalse((OUTPUT_ROOT / "smoke").exists())
        self.assertFalse((OUTPUT_ROOT / "pilot").exists())

    def test_gpu_traces_recompute_if_generated(self):
        summary_path = OUTPUT_ROOT / "summary.json"
        if not summary_path.exists():
            self.skipTest("not finalized")
        summary = json.loads(summary_path.read_text())
        if not summary["simulator_data_generated"]:
            self.skipTest("GPU stages correctly not generated")
        for phase in ("smoke", "pilot"):
            for task in TASKS:
                report_path = OUTPUT_ROOT / phase / task / "generation_report.json"
                if not report_path.exists():
                    continue
                report = json.loads(report_path.read_text())
                self.assertTrue(report["integrity"]["checks"]["trace_utility_recomputes"])
                self.assertTrue(report["integrity"]["checks"]["trace_success_recomputes"])
                with np.load(OUTPUT_ROOT / phase / task / "dataset.npz", allow_pickle=False) as data:
                    required = {
                        "history_trace_public", "history_trace_action", "candidate_trace_response_pos",
                        "candidate_trace_response_vel", "candidate_trace_actuator_pos",
                        "candidate_trace_target_pos", "candidate_trace_distance", "candidate_trace_contact",
                        "first_success_step", "initial_state_hash", "decision_state_hash", "final_state_hash",
                    }
                    self.assertTrue(required <= set(data.files))
                    recomputed = data["candidate_trace_distance"].min(axis=3)
                    self.assertTrue(np.allclose(data["utility"], -recomputed, atol=PROTOCOL["trace_recompute_tolerance"]))
                    self.assertTrue(np.array_equal(data["success"], recomputed <= PROTOCOL["success_radius"]))

    def test_final_manifests_and_authorization_if_present(self):
        summary_path = OUTPUT_ROOT / "summary.json"
        if not summary_path.exists():
            self.skipTest("not finalized")
        summary = json.loads(summary_path.read_text())
        source = json.loads((OUTPUT_ROOT / "source_manifest.json").read_text())
        for relative, record in source["files"].items():
            path = PROJECT_ROOT / relative
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        generated = json.loads((OUTPUT_ROOT / "generated_manifest.json").read_text())
        for relative, record in generated["files"].items():
            path = PROJECT_ROOT / relative
            self.assertEqual(path.stat().st_size if not path.is_symlink() else len(str(path.readlink()).encode()), record["bytes"])
            payload = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
        self.assertLessEqual(summary["candidate_executions"], PROTOCOL["candidate_execution_budget"])
        actual_bytes = sum(path.lstat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file() or path.is_symlink())
        self.assertLessEqual(actual_bytes, PROTOCOL["output_byte_budget"])
        if summary["learned_training_authorized"]:
            self.assertEqual(summary["gate"], "BINDING_AUDIT_PASS")
            self.assertTrue(summary["formal_audit"]["passed"])
        self.assertEqual(summary["learned_models_trained"], [])


if __name__ == "__main__":
    unittest.main()
