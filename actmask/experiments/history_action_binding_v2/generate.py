"""Fail-closed simulator generator entry point for the frozen v2 protocol.

The official symbolic preflight failed.  The preregistration requires an
immediate stop before GPU smoke/pilot, so this executable intentionally refuses
generation and records the exact preflight blockers instead of bypassing them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actmask.experiments.history_action_binding_v2.common import OUTPUT_ROOT, TASKS, write_json


class SymbolicPreflightBlocked(RuntimeError):
    pass


def preflight_status() -> dict:
    path = OUTPUT_ROOT / "symbolic_preflight/summary.json"
    if not path.exists():
        raise SymbolicPreflightBlocked("symbolic preflight evidence is missing")
    return json.loads(path.read_text())


def generation_refusal(phase: str, task: str, output: Path | None = None) -> dict:
    if phase not in ("smoke", "pilot"):
        raise ValueError("phase must be smoke or pilot")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    preflight = preflight_status()
    record = {
        "status": "SYMBOLIC_PREFLIGHT_FAIL",
        "generation_started": False,
        "gpu_environment_created": False,
        "candidate_executions": 0,
        "phase": phase,
        "task": task,
        "reason": "The official symbolic preflight did not pass; preregistration forbids GPU generation.",
        "failed_checks": {
            name: [key for key, passed in value["checks"].items() if not passed]
            for name, value in preflight["tasks"].items()
        },
        "recommended_fix": "Start a separately versioned v2.1 goal; do not mutate or rerun this frozen v2 protocol.",
    }
    target = OUTPUT_ROOT / "generator_refusal.json" if output is None else Path(output)
    write_json(target, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = generation_refusal(args.phase, args.task, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(3)


if __name__ == "__main__":
    main()

