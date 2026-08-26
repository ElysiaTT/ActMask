import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_icra2027_wav_external_audit.py"
SPEC = importlib.util.spec_from_file_location("icra2027_wav_external_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _base_transition():
    frame = np.zeros((6, 6, 3), dtype=np.int64)
    frame[..., 0] = 1
    frame[2, 2] = [10, 0, 0]
    return frame, np.array([5, 1], dtype=np.int64)


def test_evidence_controls_and_split_are_exact():
    n = 70
    states = np.zeros((n, 6, 6, 3), dtype=np.int64)
    next_states = np.ones((n, 6, 6, 3), dtype=np.int64)
    states[..., 0] = 1
    next_states[..., 0] = 2
    carried = np.tile(np.array([[5, 1]]), (n, 1))
    next_carried = np.tile(np.array([[0, 5]]), (n, 1))
    actions = np.repeat(np.arange(7), 10)
    data = {
        "states": states,
        "next_states": next_states,
        "carried": carried,
        "next_carried": next_carried,
        "actions": actions,
    }
    train, val = MODULE.stratified_train_val(actions, 9102)
    assert len(train) == 56
    assert len(val) == 14
    assert set(train).isdisjoint(set(val))
    assert sorted(np.bincount(actions[val], minlength=7).tolist()) == [2] * 7

    current_frame, current_carried = MODULE.select_evidence(data, "current")
    next_frame, next_carry = MODULE.select_evidence(data, "next")
    paired_frame, paired_carry = MODULE.select_evidence(data, "paired")
    assert current_frame.shape == next_frame.shape == paired_frame.shape == (n, 42, 6, 6)
    assert current_carried.shape == next_carry.shape == paired_carry.shape == (n, 34)
    assert np.array_equal(current_frame[:, :21], current_frame[:, 21:])
    assert np.array_equal(next_frame[:, :21], next_frame[:, 21:])
    assert np.array_equal(paired_frame[:, :21], current_frame[:, :21])
    assert np.array_equal(paired_frame[:, 21:], next_frame[:, 21:])


def test_transition_rule_decodes_all_seven_actions():
    current, empty = _base_transition()
    examples = []

    nxt = current.copy()
    nxt[2, 2, 2] = 3
    examples.append((current, nxt, empty, empty, 0))

    nxt = current.copy()
    nxt[2, 2, 2] = 1
    examples.append((current, nxt, empty, empty, 1))

    nxt = current.copy()
    nxt[2, 2] = [1, 0, 0]
    nxt[3, 2] = [10, 0, 0]
    examples.append((current, nxt, empty, empty, 2))

    pickup_current = current.copy()
    pickup_current[3, 2] = [5, 0, 0]
    pickup_next = pickup_current.copy()
    pickup_next[3, 2] = [1, 0, 0]
    examples.append((pickup_current, pickup_next, empty, np.array([0, 5]), 3))

    carried_key = np.array([0, 5], dtype=np.int64)
    drop_next = current.copy()
    drop_next[3, 2] = [5, 0, 0]
    examples.append((current, drop_next, carried_key, empty, 4))

    toggle_current = current.copy()
    toggle_current[3, 2] = [5, 0, 0]
    toggle_next = toggle_current.copy()
    toggle_next[3, 2, 1] = 2
    examples.append((toggle_current, toggle_next, empty, empty, 5))

    swap_current = current.copy()
    swap_current[3, 2] = [6, 1, 0]
    swap_next = swap_current.copy()
    swap_next[3, 2] = [5, 0, 0]
    examples.append((swap_current, swap_next, carried_key, np.array([1, 6]), 6))

    observed = [MODULE.transition_rule_one(s, n, c, nc) for s, n, c, nc, _ in examples]
    assert observed == list(range(7))


def test_frozen_decision_requires_every_complexity_gate():
    def summary(values):
        return {
            "by_complexity": {
                str(c): {"dynamic_accuracy": {"mean": value}}
                for c, value in zip(MODULE.COMPLEXITIES, values)
            }
        }

    summaries = {
        "current_only": summary([0.4] * 5),
        "next_only": summary([0.3] * 5),
        "paired": summary([0.8] * 5),
        "sparse_idm": summary([0.95] * 5),
        "transition_rule": summary([0.97] * 5),
    }
    assert MODULE.evaluate_decision(summaries)["pass"] is True
    summaries["paired"]["by_complexity"]["14"]["dynamic_accuracy"]["mean"] = 0.49
    decision = MODULE.evaluate_decision(summaries)
    assert decision["pass"] is False
    assert any(not gate["pass"] for gate in decision["gates"])


def test_frozen_external_report_passes_independent_verifier(tmp_path):
    output = tmp_path / "verification.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_icra2027_wav_external_audit.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert len(report["checks"]) == 9
    assert all(row["status"] == "PASS" for row in report["checks"])
