"""Validation-only selection for the single 3R learned temporal recipe."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROBUST_VARIANTS = (
    "gaussian_severe", "heteroscedastic_medium", "timestamp_jitter",
    "random_dropout", "last_frame_dropout", "burst_occlusion",
    "observation_latency", "partial_coordinates",
)


def select(report_path: str | Path, output_path: str | Path | None = None) -> dict:
    """Select a global recipe exclusively from held-out validation records."""
    report_path = Path(report_path)
    records = json.loads(report_path.read_text())["records"]
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    last: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        if row["split"] != "val" or row["variant"] not in ROBUST_VARIANTS:
            continue
        key = (row["condition"], row["method"])
        values[key].append(row["pair_order_accuracy"])
        if row["variant"] == "last_frame_dropout":
            last[key].append(row["pair_order_accuracy"])
    if not values:
        raise ValueError("no validation records for the fixed robustness suite")
    summary = [
        dict(condition=c, method=m, validation_mean=sum(v) / len(v),
             last_frame_dropout_mean=sum(last[(c, m)]) / len(last[(c, m)]),
             observations=len(v))
        for (c, m), v in values.items()
    ]
    summary.sort(key=lambda x: (-x["validation_mean"], -x["last_frame_dropout_mean"], x["condition"], x["method"]))
    result = dict(
        protocol="fixed robust-suite validation-only global selection",
        robust_variants=list(ROBUST_VARIANTS), selected=summary[0], contenders=summary,
        source=str(report_path),
    )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
