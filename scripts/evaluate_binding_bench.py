"""Evaluate external prediction directories against a frozen binding suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binding_bench.evaluator import evaluate_suite, write_evaluation


def _method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be METHOD_ID=PATH")
    method_id, path = value.split("=", 1)
    if not method_id or not path:
        raise argparse.ArgumentTypeError("method must be METHOD_ID=PATH")
    return method_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--method", action="append", type=_method, required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--baseline-method", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method_dirs = dict(args.method)
    if len(method_dirs) != len(args.method):
        raise SystemExit("duplicate --method ID")
    result = evaluate_suite(
        args.suite,
        method_dirs,
        candidate_method=args.candidate_method,
        baseline_methods=tuple(args.baseline_method),
    )
    write_evaluation(args.output, result)
    print(json.dumps({"decision": result["decision"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
