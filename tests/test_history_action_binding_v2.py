from __future__ import annotations

import hashlib
import inspect
import json
import os
import unittest
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.audit import centered_twin_preference
from actmask.experiments.history_action_binding_v2.common import OUTPUT_ROOT, PROTOCOL, TASKS
from actmask.experiments.history_action_binding_v2.generate import generation_refusal
from actmask.experiments.history_action_binding_v2.run_v2 import tree_manifest, verify_other_protected, verify_v1_tree
from actmask.experiments.history_action_binding_v2.symbolic import candidate_sampler, symbolic_dataset


class ProtocolTest(unittest.TestCase):
    def test_candidate_sampler_public_inputs_only_and_deterministic(self):
        self.assertEqual(tuple(inspect.signature(candidate_sampler).parameters), ("task", "public_rotation", "candidate_seed"))
        source = inspect.getsource(candidate_sampler).lower()
        for forbidden in ("latent_phi", "alpha", "beta", "branch", "future", "label", "success"):
            self.assertNotIn(forbidden, source)
        one = candidate_sampler(TASKS[0], 0.37, 12345)
        two = candidate_sampler(TASKS[0], 0.37, 12345)
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(one, two)))

    def test_symbolic_marginals_and_candidates_exact(self):
        for task in TASKS:
            data = symbolic_dataset(task, 64, PROTOCOL["seed"])
            self.assertTrue(np.array_equal(data["probe_actions"][:, 0], data["probe_actions"][:, 1]))
            self.assertTrue(np.array_equal(np.sort(data["probe_delta"][:, 0], axis=1), np.sort(data["probe_delta"][:, 1], axis=1)))
            self.assertTrue(np.array_equal(np.sort(data["probe_utility"][:, 0], axis=1), np.sort(data["probe_utility"][:, 1], axis=1)))
            self.assertTrue(np.array_equal(data["candidate_actions_by_branch"][:, 0], data["candidate_actions_by_branch"][:, 1]))

    def test_novel_magnitudes(self):
        probes = set(PROTOCOL["task_b_probe_magnitudes"])
        candidates = set(PROTOCOL["candidate_magnitudes"])
        self.assertTrue(probes.isdisjoint(candidates))

    def test_centered_metric_rejects_branch_constants(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(20, 2, 16))
        constants = np.broadcast_to(np.asarray([[-1.0], [2.0]])[None], truth.shape)
        templates = np.tile(np.arange(16), (20, 1))
        score, *_ = centered_twin_preference(truth, constants, templates)
        self.assertEqual(score, 0.5)


class FrozenTerminalEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = OUTPUT_ROOT / "summary.json"
        cls.summary = json.loads(path.read_text()) if path.exists() else None

    def test_official_preflight_failed_without_gpu_data(self):
        preflight = json.loads((OUTPUT_ROOT / "symbolic_preflight/summary.json").read_text())
        self.assertFalse(preflight["passed"])
        self.assertFalse((OUTPUT_ROOT / "smoke").exists())
        self.assertFalse((OUTPUT_ROOT / "pilot").exists())
        self.assertIn("candidate_slots_balanced", [key for key, value in preflight["tasks"][TASKS[0]]["checks"].items() if not value])
        self.assertIn("fourier_at_least_0_90", [key for key, value in preflight["tasks"][TASKS[0]]["checks"].items() if not value])
        self.assertAlmostEqual(preflight["tasks"][TASKS[0]]["fourier_score"], 0.87890625)
        self.assertAlmostEqual(preflight["tasks"][TASKS[1]]["fourier_score"], 0.908203125)

    def test_marginal_and_arrow_repairs_are_visible_but_not_overclaimed(self):
        preflight = json.loads((OUTPUT_ROOT / "symbolic_preflight/summary.json").read_text())
        for task in TASKS:
            item = preflight["tasks"][task]
            self.assertEqual(max(item["marginal_scores"].values()), 0.5)
            self.assertEqual(item["arrow_score"], 0.5)
            self.assertEqual(item["candidate_top1_crossing_rate"], 1.0)
            self.assertTrue(item["checks"]["result_multiset_exact"])

    def test_all_symbolic_interventions_saved(self):
        for task in TASKS:
            path = OUTPUT_ROOT / "symbolic_preflight" / task / "intervention_samples.npz"
            self.assertTrue(path.exists())
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "original_probe_actions", "pair_preserving_probe_actions",
                    "binding_break_probe_actions", "twin_swap_probe_actions",
                    "fixed_current_observation", "fixed_candidate_actions",
                    "slot_permuted_candidates", "rotated_probe_actions",
                    "simulator_expected_preference", "expected_preference_after_twin_swap",
                }
                self.assertTrue(required <= set(data.files))
                self.assertTrue(np.array_equal(data["original_probe_actions"], data["binding_break_probe_actions"]))
                self.assertTrue(np.array_equal(
                    np.sort(data["original_probe_results"], axis=2),
                    np.sort(data["binding_break_probe_results"], axis=2),
                ))

    def test_generator_fails_closed(self):
        target = OUTPUT_ROOT / "test_generator_refusal.json"
        record = generation_refusal("smoke", TASKS[0], target)
        self.assertFalse(record["generation_started"])
        self.assertEqual(record["candidate_executions"], 0)
        target.unlink()

    def test_postmortem_and_protected_evidence(self):
        postmortem = json.loads((OUTPUT_ROOT / "v1_failure_postmortem.json").read_text())
        self.assertEqual(postmortem["v1_gate"], "SHORTCUT_SATURATED")
        self.assertFalse(postmortem["learned_training_was_authorized"])
        self.assertTrue(verify_v1_tree()["passed"])
        self.assertTrue(verify_other_protected()["passed"])

    def test_terminal_summary_never_authorizes_training(self):
        if self.summary is None:
            self.skipTest("terminal summary not finalized")
        self.assertEqual(self.summary["gate"], "SYMBOLIC_PREFLIGHT_FAIL")
        self.assertFalse(self.summary["simulator_data_generated"])
        self.assertFalse(self.summary["binding_necessity_established"])
        self.assertFalse(self.summary["learned_training_authorized"])
        self.assertEqual(self.summary["candidate_executions"], 0)
        self.assertLessEqual(self.summary["output_bytes"], PROTOCOL["output_byte_budget"])

    def test_saved_manifests_exact(self):
        if self.summary is None:
            self.skipTest("terminal summary not finalized")
        source = json.loads((OUTPUT_ROOT / "source_manifest.json").read_text())
        for relative, record in source["files"].items():
            path = Path(__file__).resolve().parents[1] / relative
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        saved = json.loads((OUTPUT_ROOT / "generated_manifest.json").read_text())
        fresh = tree_manifest(OUTPUT_ROOT, exclude={OUTPUT_ROOT / "generated_manifest.json"})
        self.assertEqual(saved, fresh)


if __name__ == "__main__":
    unittest.main()
