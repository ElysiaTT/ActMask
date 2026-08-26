"""Add independently recomputed short/medium/long RM-SPD metrics from raw files."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from actmask.experiments.rm_spd_baselines import _metric_one
from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, write_json, write_jsonl


HORIZONS = ("short", "medium", "long")
NUMERIC = ("scene_cosine_error", "scene_l2_error", "patch_retrieval_accuracy", "change_magnitude_error", "static_border_error_proxy", "robot_core_l2_diagnostic", "uncertain_region_coverage", "weighted_l2", "full_frame_l2")


def run() -> dict[str, int]:
    raw = list(iter_jsonl(OUT / "raw_prediction_manifest.json")); rows = []
    for record in raw:
        z = np.load(OUT / record["prediction_path"], allow_pickle=False)
        for index, horizon in enumerate(HORIZONS):
            metric = _metric_one(z["predicted_residual"][index:index + 1].astype(np.float32), z["target_residual"][index:index + 1].astype(np.float32), z["target_feature"][index:index + 1].astype(np.float32), z["region_state"][index:index + 1], z["loss_weight"][index:index + 1].astype(np.float32))
            rows.append({**{k: record[k] for k in ("model", "regime", "sample_id", "task_id", "episode_split", "task_split")}, "horizon": horizon, **metric})
    write_jsonl(OUT / "baseline_per_horizon_metrics.jsonl", rows)
    summary_doc = json.loads((OUT / "baseline_summary.json").read_text()); grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows: grouped.setdefault((row["model"], row["regime"], row["horizon"]), []).append(row)
    for key, value in summary_doc["summary"].items():
        model, regime = key.split("::"); horizon_summary = {}
        for horizon in HORIZONS:
            values = grouped[(model, regime, horizon)]
            metric_dict = {}
            for metric_name in NUMERIC:
                x = [r[metric_name] for r in values if not math.isnan(r[metric_name])]
                metric_dict[metric_name] = float(np.mean(x)) if x else float("nan")
            horizon_summary[horizon] = metric_dict
        value["per_horizon"] = horizon_summary
    summary_doc["per_horizon_metric_source"] = "baseline_per_horizon_metrics.jsonl; recomputed from raw predictions after training"
    write_json(OUT / "baseline_summary.json", summary_doc)
    write_json(OUT / "baseline_horizon_audit.json", {"schema": "rm-spd-per-horizon-audit-v1", "raw_prediction_records": len(raw), "per_horizon_records": len(rows), "horizons": list(HORIZONS), "source": "raw_prediction_manifest.json", "all_three_horizons_present": len(rows) == 3 * len(raw)})
    append_log("S5_PER_HORIZON_AUDIT_COMPLETE", raw_predictions=len(raw), records=len(rows))
    return {"raw_predictions": len(raw), "per_horizon_records": len(rows)}


if __name__ == "__main__": print(json.dumps(run(), sort_keys=True))
