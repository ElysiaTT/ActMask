"""Validation-selected analytic baseline and grouped final 3R comparison."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from actmask.experiments.milestone3r_metrics import paired_bootstrap
from actmask.experiments.milestone3r_selection import ROBUST_VARIANTS


def _mean(rows):
    return float(np.mean([x["pair_order_accuracy"] if isinstance(x, dict) else x for x in rows]))


def compare(analytic_path: str | Path, learned_path: str | Path, output_path: str | Path | None = None) -> dict:
    analytic = json.loads(Path(analytic_path).read_text())["records"]
    learned = json.loads(Path(learned_path).read_text())["records"]
    a_val = defaultdict(list)
    for row in analytic:
        if row["split"] == "val" and row.get("fair", True) and row["variant"] in ROBUST_VARIANTS:
            a_val[row["method"]].append(row["pair_order_accuracy"])
    selected_analytic = sorted(a_val, key=lambda k: (-_mean(a_val[k]), k))[0]
    variants = sorted({r["variant"] for r in learned if r["split"] == "test"})
    table = []
    deltas, groups, robust_deltas, robust_groups = [], [], [], []
    for variant in variants:
        learned_rows = [r for r in learned if r["split"] == "test" and r["variant"] == variant]
        analytic_rows = [r for r in analytic if r["split"] == "test" and r["variant"] == variant and r["method"] == selected_analytic]
        by_task = {task: _mean([r for r in learned_rows if r["task"] == task]) for task in {r["task"] for r in learned_rows}}
        a_by_task = {r["task"]: r["pair_order_accuracy"] for r in analytic_rows}
        difference = {task: by_task[task] - a_by_task[task] for task in by_task}
        table.append(dict(variant=variant, learned_mean=_mean(learned_rows), analytic_mean=_mean(analytic_rows), delta=float(np.mean(list(difference.values())))))
        for task, delta in difference.items():
            deltas.append(delta); groups.append(f"{variant}:{task}")
            if variant in ROBUST_VARIANTS:
                robust_deltas.append(delta); robust_groups.append(f"{variant}:{task}")
    robust = [x for x in table if x["variant"] in ROBUST_VARIANTS]
    result = dict(
        analytic_validation_means={k: _mean(v) for k, v in a_val.items()},
        selected_analytic=selected_analytic,
        selected_learned=dict(condition="clean", method="gru", seeds=[17,29,43,59,71]),
        test_by_variant=table,
        robust_test_mean=dict(learned=float(np.mean([x["learned_mean"] for x in robust])), analytic=float(np.mean([x["analytic_mean"] for x in robust])), delta=float(np.mean([x["delta"] for x in robust]))),
        grouped_bootstrap=paired_bootstrap(np.asarray(robust_deltas), np.asarray(robust_groups)),
        all_variant_grouped_bootstrap=paired_bootstrap(np.asarray(deltas), np.asarray(groups)),
        bootstrap_unit="task-by-corruption mean over the five learned seeds",
    )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
