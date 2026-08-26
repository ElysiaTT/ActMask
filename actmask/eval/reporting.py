"""Deterministic JSON, CSV, Markdown, and five-seed reporting helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPORT_SORT_KEYS = (
    "distribution",
    "ood_axis",
    "counterfactual_type",
    "scenario",
    "method_id",
    "training_variant",
    "eval_perturbation",
    "row_type",
    "statistic",
    "seed",
)

DEFAULT_FIELD_ORDER = (
    "row_type",
    "statistic",
    "seed",
    "seed_count",
    "method_id",
    "method_kind",
    "training_variant",
    "eval_perturbation",
    "split",
    "distribution",
    "ood_axis",
    "counterfactual_type",
    "scenario",
    "threshold",
    "threshold_source",
    "samples",
    "points",
    "positives",
    "negatives",
    "positive_prevalence",
    "has_positive_labels",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "iou",
    "average_precision",
    "pr_auc",
    "paired_consistency",
    "pair_count",
    "success_ranking_accuracy",
    "ranking_pair_count",
    "change_iou",
    "change_ap",
    "change_pr_auc",
    "signed_direction_accuracy",
    "changed_region_probability_delta",
    "invariant_probability_mae",
    "invariant_probability_stability",
    "invariant_binary_agreement",
    "invariant_prediction_iou",
    "changed_background_false_positive_rate",
)

DEFAULT_GROUP_KEYS = (
    "method_id",
    "method_kind",
    "training_variant",
    "eval_perturbation",
    "split",
    "distribution",
    "ood_axis",
    "counterfactual_type",
    "scenario",
    "threshold_source",
)

DEFAULT_METRIC_KEYS = (
    "precision",
    "recall",
    "f1",
    "iou",
    "accuracy",
    "average_precision",
    "pr_auc",
    "positive_prevalence",
    "positive_mask_prevalence",
    "paired_consistency",
    "paired_scene_consistency",
    "success_ranking_accuracy",
    "change_iou",
    "change_ap",
    "change_pr_auc",
    "signed_direction_accuracy",
    "signed_response_accuracy",
    "changed_region_probability_delta",
    "invariant_probability_mae",
    "invariant_probability_stability",
    "invariant_binary_agreement",
    "invariant_prediction_iou",
    "changed_background_false_positive_rate",
)

SUPPORT_KEYS = (
    "samples",
    "points",
    "positives",
    "negatives",
    "has_positive_labels",
    "tp",
    "fp",
    "fn",
    "tn",
    "pair_count",
    "paired_comparisons",
    "ranking_pair_count",
    "ranking_comparisons",
    "ground_truth_changed_points",
    "invariant_points",
    "changed_background_points",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sort_token(value: Any) -> tuple[int, str]:
    return (1, "") if value is None else (0, str(value))


_STATISTIC_ORDER = {"single": 0, "mean": 1, "std": 2, "min": 3, "max": 4}


def _record_sort_token(key: str, value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    if key == "statistic":
        return (0, (_STATISTIC_ORDER.get(str(value), 99), str(value)))
    if key == "seed" and isinstance(value, (int, np.integer)):
        return (0, int(value))
    return (1, str(value))


def sort_records(
    records: Iterable[Mapping[str, Any]], sort_keys: Sequence[str] = REPORT_SORT_KEYS
) -> list[dict[str, Any]]:
    normalized = [dict(record) for record in records]
    return sorted(
        normalized,
        key=lambda record: tuple(
            _record_sort_token(key, record.get(key)) for key in sort_keys
        ),
    )


def write_json_report(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema_version: str = "2.0",
    sort_keys: Sequence[str] = REPORT_SORT_KEYS,
) -> Path:
    """Write a byte-stable report containing metadata and sorted records."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": str(schema_version),
        "experiment": _json_safe(dict(metadata or {})),
        "records": _json_safe(sort_records(records, sort_keys)),
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output_path


def _fieldnames(records: list[Mapping[str, Any]]) -> list[str]:
    keys = {key for record in records for key in record}
    ordered = [key for key in DEFAULT_FIELD_ORDER if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def _csv_cell(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, bool):
        return "true" if safe else "false"
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe


def write_csv_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
    *,
    sort_keys: Sequence[str] = REPORT_SORT_KEYS,
) -> Path:
    """Write sorted records with a deterministic union-of-fields header."""

    sorted_rows = sort_records(records, sort_keys)
    if not sorted_rows:
        raise ValueError("cannot write an empty CSV report")
    columns = list(fieldnames) if fieldnames is not None else _fieldnames(sorted_rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in columns})
    return output_path


def seed_summary_records(
    records: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    seed_key: str = "seed",
    expected_seed_count: int | None = 5,
) -> list[dict[str, Any]]:
    """Return mean/std/min/max rows across learned deterministic seeds.

    Standard deviation is the sample standard deviation (``ddof=1``).  Support
    and confusion counts are set to ``None`` in summary rows rather than being
    misleadingly averaged.  Records without a seed (normally deterministic
    baselines) are ignored.
    """

    if expected_seed_count is not None and expected_seed_count < 2:
        raise ValueError("expected_seed_count must be at least 2 or None")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        record = dict(source)
        if record.get(seed_key) is None:
            continue
        groups[tuple(record.get(key) for key in group_keys)].append(record)

    output: list[dict[str, Any]] = []
    statistic_functions = {
        "mean": lambda values: float(np.mean(values)),
        "std": lambda values: float(np.std(values, ddof=1)),
        "min": lambda values: float(np.min(values)),
        "max": lambda values: float(np.max(values)),
    }
    for _, group in sorted(
        groups.items(), key=lambda item: tuple(_sort_token(value) for value in item[0])
    ):
        seeds = [record[seed_key] for record in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds within one summary group: {seeds}")
        if expected_seed_count is not None and len(seeds) != expected_seed_count:
            raise ValueError(
                f"expected {expected_seed_count} seeds per group, found {len(seeds)}: {seeds}"
            )
        metrics: dict[str, np.ndarray] = {}
        for key in metric_keys:
            values = [record.get(key) for record in group]
            if all(value is not None and not isinstance(value, bool) for value in values):
                try:
                    numeric = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric).all():
                    metrics[key] = numeric

        template = dict(group[0])
        for statistic, operation in statistic_functions.items():
            row = dict(template)
            row["row_type"] = "seed_summary"
            row["statistic"] = statistic
            row[seed_key] = None
            row["seed_count"] = len(seeds)
            row["threshold"] = None
            for key in SUPPORT_KEYS:
                if key in row:
                    row[key] = None
            for key in metric_keys:
                row[key] = operation(metrics[key]) if key in metrics else None
            output.append(row)
    return sort_records(output)


# Concise orchestration alias.
summarize_seeds = seed_summary_records


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_summary_markdown(
    path: str | Path,
    scenario_records: Iterable[Mapping[str, Any]],
    counterfactual_records: Iterable[Mapping[str, Any]] | None = None,
    *,
    title: str = "ActMask Milestone 2A summary",
    strongest_baseline: str | None = None,
) -> Path:
    """Write deterministic readable tables directly from metric records."""

    scenarios = sort_records(scenario_records)
    counterfactuals = sort_records(counterfactual_records or [])
    lines = [f"# {title}", ""]
    if strongest_baseline is not None:
        lines.extend([f"Strongest validation-selected geometric baseline: `{strongest_baseline}`.", ""])
    lines.extend(
        [
            "## Scenario-wise metrics",
            "",
            "| Distribution | OOD axis | Scenario | Method | Seed/stat | IoU | F1 | AP | PR-AUC | Pair consistency | Success ranking |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scenarios:
        seed_label = row.get("seed")
        if seed_label is None:
            seed_label = row.get("statistic", "single")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("distribution", "—")),
                    str(row.get("ood_axis", "—")),
                    str(row.get("scenario", "—")),
                    str(row.get("method_id", "—")),
                    str(seed_label),
                    _format_metric(row.get("iou")),
                    _format_metric(row.get("f1")),
                    _format_metric(row.get("average_precision")),
                    _format_metric(row.get("pr_auc")),
                    _format_metric(row.get("paired_consistency")),
                    _format_metric(row.get("success_ranking_accuracy")),
                )
            )
            + " |"
        )

    if counterfactuals:
        lines.extend(
            [
                "",
                "## Counterfactual consistency",
                "",
                "| Type | Scenario | Method | Seed/stat | Change IoU | Change AP | Signed response | Invariant MAE | Binary agreement | Background FPR |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in counterfactuals:
            seed_label = row.get("seed")
            if seed_label is None:
                seed_label = row.get("statistic", "single")
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row.get("counterfactual_type", "—")),
                        str(row.get("scenario", "—")),
                        str(row.get("method_id", "—")),
                        str(seed_label),
                        _format_metric(row.get("change_iou")),
                        _format_metric(row.get("change_ap")),
                        _format_metric(row.get("signed_direction_accuracy")),
                        _format_metric(row.get("invariant_probability_mae")),
                        _format_metric(row.get("invariant_binary_agreement")),
                        _format_metric(row.get("changed_background_false_positive_rate")),
                    )
                )
                + " |"
            )
    lines.append("")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output_path


__all__ = [
    "seed_summary_records",
    "sort_records",
    "summarize_seeds",
    "write_csv_records",
    "write_json_report",
    "write_summary_markdown",
]
