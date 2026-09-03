"""Generate and evaluate the small CPU-only binding benchmark example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binding_bench.example import run_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit/results/binding_bench_example"),
    )
    args = parser.parse_args()
    result = run_example(args.output_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "strongest_baseline": result["strongest_baseline"],
                "comparison": result["comparison"],
                "intervention_effects": result["intervention_effects"],
                "checks": result["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
