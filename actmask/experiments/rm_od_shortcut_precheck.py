"""Pre-registered source/candidate controls before fair RM-OD baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from actmask.experiments.rm_od_freeze_and_inventory import OUT, append_log, write_json
from actmask.experiments.rm_od_metrics import group_metrics, score_rows


def _write(name: str, family: str, partition: str, rows: list[dict]) -> dict:
    path = OUT / "raw_scores" / "precheck" / family / partition / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {"path": str(path.relative_to(OUT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metrics": group_metrics(rows)}


def run() -> dict:
    groups = [json.loads(line) for line in (OUT / "task_groups.jsonl").read_text().splitlines() if line]
    report: dict[str, dict] = {}
    entries = []
    for family in sorted({group["split_family"] for group in groups}):
        report[family] = {}
        for partition in ("train", "validation", "test"):
            current = [group for group in groups if group["split_family"] == family and group["partition"] == partition]
            controls = {
                "constant": lambda group: [0.0] * len(group["candidates"]),
                "candidate_index": lambda group: [-candidate["candidate_index"] for candidate in group["candidates"]],
                "source_path_full": lambda group: [1.0 if candidate["candidate_source_provenance"]["parquet"] == group["anchor_source_provenance"]["parquet"] else 0.0 for candidate in group["candidates"]],
                "camera_background": lambda group: [0.0] * len(group["candidates"]),
            }
            report[family][partition] = {}
            for name, scorer in controls.items():
                rows = score_rows(current, scorer)
                entry = _write(name, family, partition, rows)
                report[family][partition][name] = entry["metrics"]
                entries.append({"family": family, "partition": partition, "name": name, **entry})
    source_test = max(report[family]["test"]["source_path_full"]["recall_at_1"] for family in report)
    decision = "RM_OD_ACTION_PATH_SHORTCUT" if source_test > 0.225 else "RM_OD_CONTROLS_PASS"
    write_json(OUT / "initial_shortcut_precheck.json", {"schema": "rm-od-initial-shortcut-precheck-v1", "report": report, "entries": entries, "random_recall_at_1": 0.125, "shortcut_limit": 0.225, "decision": decision, "reason": "full candidate source path is a diagnostic only and never a fair model input"})
    append_log("phase5_initial_shortcut_precheck", decision=decision, source_path_test_recall_at_1=source_test)
    return {"decision": decision, "source_path_test_recall_at_1": source_test}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
