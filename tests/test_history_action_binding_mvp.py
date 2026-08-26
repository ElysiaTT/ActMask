from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding.audit import (
    BASELINES,
    baseline_predictions,
    binding_breaking_shuffle,
    candidate_slot_permutation,
    coordinate_reflection,
    metrics,
    pair_preserving_shuffle,
    twin_history_swap,
)
from actmask.experiments.history_action_binding.common import OUTPUT_ROOT, PROTOCOL, split_for_twin
from actmask.experiments.history_action_binding.generate import (
    _candidate_arrays,
    _latents,
    candidate_sampler,
)


class HistoryActionBindingProtocolTest(unittest.TestCase):
    def test_candidate_sampler_is_latent_independent_and_deterministic(self):
        self.assertEqual(tuple(inspect.signature(candidate_sampler).parameters), ("candidate_seed",))
        source = inspect.getsource(candidate_sampler)
        for forbidden in ("latent", "mode", "alpha", "beta", "label", "success", "future", "twin"):
            self.assertNotIn(forbidden, source.lower())
        left, left_order = candidate_sampler(1234)
        right, right_order = candidate_sampler(1234)
        self.assertTrue(np.array_equal(left, right))
        self.assertTrue(np.array_equal(left_order, right_order))
        self.assertEqual(left.shape, (PROTOCOL["candidates_per_twin"], PROTOCOL["candidate_horizon"], 1))

    def test_novel_candidate_magnitudes_are_disjoint(self):
        probes = set(np.abs(np.concatenate(list(PROTOCOL["probe_profiles"].values()))))
        candidates = set(np.abs(PROTOCOL["candidate_amplitudes"]))
        self.assertTrue(probes.isdisjoint(candidates))

    def test_candidate_and_latent_regeneration(self):
        for task in ("task_a_discrete_mode", "task_b_continuous_response"):
            one = _candidate_arrays(task, 8, PROTOCOL["seed"])
            two = _candidate_arrays(task, 8, PROTOCOL["seed"])
            for left, right in zip(one, two):
                self.assertTrue(np.array_equal(left, right))
            one_latent = _latents(task, 8, PROTOCOL["seed"] + 17)
            two_latent = _latents(task, 8, PROTOCOL["seed"] + 17)
            for key in one_latent:
                self.assertTrue(np.array_equal(one_latent[key], two_latent[key]))

    def test_split_is_twin_disjoint_and_seed_independent(self):
        groups = {name: set() for name in ("train", "validation", "test")}
        for twin in range(PROTOCOL["pilot_twins_per_task"]):
            groups[split_for_twin(twin)].add(twin)
        self.assertTrue(groups["train"])
        self.assertTrue(groups["validation"])
        self.assertTrue(groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_interventions_preserve_exact_contract(self):
        actions = np.arange(2 * 2 * 6, dtype=np.float32).reshape(2, 2, 6)
        results = (100 + np.arange(2 * 2 * 6, dtype=np.float32)).reshape(2, 2, 6)
        pair_action, pair_result = pair_preserving_shuffle(actions, results)
        for twin in range(2):
            for branch in range(2):
                original_pairs = sorted(zip(actions[twin, branch].tolist(), results[twin, branch].tolist()))
                shuffled_pairs = sorted(zip(pair_action[twin, branch].tolist(), pair_result[twin, branch].tolist()))
                self.assertEqual(original_pairs, shuffled_pairs)
        broken_action, broken_result = binding_breaking_shuffle(actions, results)
        self.assertTrue(np.array_equal(broken_action, actions))
        self.assertFalse(np.array_equal(broken_result, results))
        self.assertTrue(np.array_equal(np.sort(broken_result, axis=2), np.sort(results, axis=2)))
        swapped_action, swapped_result = twin_history_swap(actions, results)
        self.assertTrue(np.array_equal(swapped_action[:, 0], actions[:, 1]))
        self.assertTrue(np.array_equal(swapped_result[:, 1], results[:, 0]))


class FrozenEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = []
        for phase in ("smoke", "pilot"):
            for task in ("task_a_discrete_mode", "task_b_continuous_response"):
                path = OUTPUT_ROOT / phase / task / "dataset.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        cls.datasets.append((phase, task, path, {key: archive[key] for key in archive.files}))

    def test_final_evidence_exists(self):
        if not (OUTPUT_ROOT / "summary.json").exists():
            self.skipTest("final evidence not generated yet")
        self.assertEqual(len(self.datasets), 4)

    def test_candidate_equality_trace_recompute_and_all_outcomes_retained(self):
        if not self.datasets:
            self.skipTest("runtime evidence not generated yet")
        for _, _, path, dataset in self.datasets:
            self.assertTrue(np.array_equal(dataset["candidate_actions_by_branch"][:, 0], dataset["candidate_actions_by_branch"][:, 1]))
            recomputed_min = dataset["candidate_trace_distance"].min(axis=3)
            recomputed_success = recomputed_min <= PROTOCOL["success_radius"]
            self.assertTrue(np.array_equal(recomputed_success, dataset["success"]))
            self.assertTrue(np.allclose(-recomputed_min, dataset["utility"], atol=PROTOCOL["trace_recompute_tolerance"], rtol=0))
            report = json.loads((path.parent / "generation_report.json").read_text())
            self.assertEqual(sum(report["integrity"]["outcome_counts"].values()), dataset["success"].shape[0] * dataset["success"].shape[2])
            self.assertEqual(report["candidate_executions"], dataset["success"].size)

    def test_matching_and_current_observation(self):
        if not self.datasets:
            self.skipTest("runtime evidence not generated yet")
        for _, _, _, dataset in self.datasets:
            self.assertLessEqual(float(np.max(np.abs(dataset["decision_physical_state"][:, :, 0]))), PROTOCOL["matching_position_tolerance"])
            self.assertLessEqual(float(np.max(np.abs(dataset["decision_physical_state"][:, :, 1]))), PROTOCOL["matching_velocity_tolerance"])
            self.assertTrue(np.array_equal(dataset["decision_public_state"][:, 0], dataset["decision_public_state"][:, 1]))

    def test_slot_and_reflection_invariance_for_explicit_fits(self):
        pilots = [entry for entry in self.datasets if entry[0] == "pilot"]
        if not pilots:
            self.skipTest("pilot evidence not generated yet")
        for _, _, _, dataset in pilots:
            original = baseline_predictions(dataset)
            permuted_dataset = candidate_slot_permutation(dataset)
            permuted = baseline_predictions(permuted_dataset)
            reflected_dataset = coordinate_reflection(dataset)
            reflected = baseline_predictions(reflected_dataset)
            for name in ("linear_least_squares_response", "quadratic_least_squares_response"):
                original_metrics = metrics(dataset, original[name])
                permuted_metrics = metrics(permuted_dataset, permuted[name])
                reflected_metrics = metrics(reflected_dataset, reflected[name])
                for key in ("ndcg", "normalized_regret", "twin_preference_accuracy"):
                    self.assertAlmostEqual(original_metrics[key], permuted_metrics[key], places=12)
                    self.assertAlmostEqual(original_metrics[key], reflected_metrics[key], places=12)

    def test_all_baselines_and_prediction_rows_are_saved(self):
        if not (OUTPUT_ROOT / "summary.json").exists():
            self.skipTest("final evidence not generated yet")
        for task in ("task_a_discrete_mode", "task_b_continuous_response"):
            summary = json.loads((OUTPUT_ROOT / "pilot" / task / "audit" / "audit_summary.json").read_text())
            self.assertEqual(set(summary["metrics"]["original"]), set(BASELINES))
            csv_path = OUTPUT_ROOT / "pilot" / task / "audit" / "baseline_predictions.csv"
            rows = sum(1 for _ in csv_path.open()) - 1
            expected = PROTOCOL["pilot_twins_per_task"] * 2 * PROTOCOL["candidates_per_twin"] * len(BASELINES) * 6
            self.assertEqual(rows, expected)

    def test_budget_and_manifest(self):
        summary_path = OUTPUT_ROOT / "summary.json"
        if not summary_path.exists():
            self.skipTest("final evidence not generated yet")
        summary = json.loads(summary_path.read_text())
        self.assertLessEqual(summary["candidate_executions"], PROTOCOL["candidate_execution_budget"])
        self.assertLessEqual(summary["output_bytes"], PROTOCOL["output_byte_budget"])
        self.assertFalse(any(summary["learned_models_trained"]))
        self.assertEqual(summary["learned_training_authorized"], summary["gate"] == "BINDING_AUDIT_PASS")
        manifest = json.loads((OUTPUT_ROOT / "generated_manifest.json").read_text())
        for relative, record in manifest["files"].items():
            path = OUTPUT_ROOT / relative
            import hashlib
            if record["kind"] == "symlink":
                import os
                target = os.readlink(path)
                self.assertEqual(len(target.encode()), record["bytes"])
                self.assertEqual(hashlib.sha256(target.encode()).hexdigest(), record["sha256"])
            else:
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
