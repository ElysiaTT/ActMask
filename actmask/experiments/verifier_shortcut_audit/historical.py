"""V1 historical import verification and V2 family role assignment."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    HISTORICAL,
    OUT,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from .preregister import FAMILIES


def _decision(branch: str, relative: str) -> dict[str, Any]:
    return read_json(HISTORICAL / branch / relative)


def _verify_config_hash(branch: str) -> dict[str, Any]:
    root = HISTORICAL / branch
    config = root / "preregistered_config.json"
    hash_file = root / "preregistered_config.sha256"
    if not config.is_file() or not hash_file.is_file():
        return {
            "branch": branch,
            "available": False,
            "pass": False,
            "reason": "preregistered config/hash absent",
        }
    actual = sha256_file(config)
    recorded = hash_file.read_text(encoding="utf-8").strip()
    return {
        "branch": branch,
        "available": True,
        "actual_sha256": actual,
        "recorded_sha256": recorded,
        "pass": actual == recorded,
    }


def _source_rows() -> list[dict[str, Any]]:
    sources = [
        (
            "rm_acv_pilot166",
            HISTORICAL / "rm_acv_pilot166/source_hash_manifest.json",
        ),
        (
            "rm_series_robomind",
            HISTORICAL / "rm_series_robomind/source_hash_manifest.json",
        ),
        (
            "rm_od_future_object_dynamics",
            HISTORICAL / "rm_od_future_object_dynamics/source_hash_manifest.json",
        ),
        (
            "rm_spd_sam3_prior_dynamics",
            HISTORICAL / "rm_spd_sam3_prior_dynamics/source_hash_manifest.json",
        ),
    ]
    by_path: dict[str, dict[str, Any]] = {}
    conflicts = []
    for branch, manifest_path in sources:
        manifest = read_json(manifest_path)
        rows = manifest.get("entries", manifest.get("files", []))
        for row in rows:
            raw_path = (
                row.get("source_path_provenance_only")
                or row.get("local_path")
                or row.get("source_path")
                or row.get("path")
            )
            if not raw_path:
                conflicts.append(
                    {
                        "branch": branch,
                        "reason": "source row has no path",
                        "row": row,
                    }
                )
                continue
            record = {
                "path": str(Path(raw_path).resolve()),
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
                "declared_by": [branch],
            }
            if record["path"] in by_path:
                old = by_path[record["path"]]
                if (old["bytes"], old["sha256"]) != (
                    record["bytes"],
                    record["sha256"],
                ):
                    conflicts.append(
                        {
                            "path": record["path"],
                            "reason": "conflicting historical hashes",
                            "existing": old,
                            "new": record,
                        }
                    )
                else:
                    old["declared_by"].append(branch)
            else:
                by_path[record["path"]] = record
    rows = sorted(by_path.values(), key=lambda row: row["path"])
    if conflicts:
        write_json(OUT / "source_hash_conflicts.json", conflicts)
    return rows


def _rehash_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    results = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        path = Path(row["path"])
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha = sha256_file(path) if exists else None
        passed = (
            exists
            and actual_bytes == row["bytes"]
            and actual_sha == row["sha256"]
        )
        result = {
            **row,
            "exists": exists,
            "actual_bytes": actual_bytes,
            "actual_sha256": actual_sha,
            "pass": passed,
        }
        results.append(result)
        if not passed:
            failures.append(result)
        if index % 25 == 0 or index == len(rows):
            print(
                f"VSA_SOURCE_REHASH={index}/{len(rows)} failures={len(failures)}",
                flush=True,
            )
    return {
        "schema": "vsa-source-content-rehash-v1",
        "unique_source_files": len(rows),
        "declared_bytes": sum(row["bytes"] for row in rows),
        "elapsed_seconds": time.time() - started,
        "failures": failures,
        "files": results,
        "pass": not failures,
    }


def _historical_inventory() -> dict[str, Any]:
    pilot_decision = _decision("rm_acv_pilot166", "final_decision.json")
    real_decision = _decision("rm_acv_realneg_v1", "final_decision.json")
    rm_decision = _decision(
        "rm_series_robomind", "final_package/final_decision.json"
    )
    od_decision = _decision("rm_od_future_object_dynamics", "final_decision.json")
    spd_decision = _decision(
        "rm_spd_sam3_prior_dynamics", "final_decision.json"
    )
    negative_rows = read_jsonl(
        HISTORICAL / "rm_acv_pilot166/negative_candidate_manifest.jsonl"
    )
    family_counts = Counter(row["family"] for row in negative_rows)
    n7 = read_jsonl(HISTORICAL / "rm_acv_realneg_v1/n7_pair_manifest.jsonl")
    n8 = read_jsonl(HISTORICAL / "rm_acv_realneg_v1/n8_pair_manifest.jsonl")
    validity = read_json(
        HISTORICAL / "rm_acv_realneg_v1/behavioral_validity_report.json"
    )
    family_decision = read_json(
        HISTORICAL / "rm_acv_pilot166/negative_family_decision.json"
    )
    return {
        "schema": "vsa-historical-branch-inventory-v1",
        "branches": [
            {
                "branch": "rm_series_robomind",
                "terminal_decision": rm_decision["decision"],
                "intended_task": "logged action-conditioned future consistency",
                "negative_construction": "historical RM-Series paired construction",
                "known_shortcut": (
                    "source/path controls measurable; state-action signal with "
                    "weaker task-held-out generalization"
                ),
                "known_ambiguity": "no outcome/failure or physical counterfactual label",
                "learned_models_trained": bool(rm_decision["method_ran"]),
                "behavioral_evidence": False,
                "permitted_role": "historical motivation and split/example audit",
            },
            {
                "branch": "rm_acv_pilot166",
                "terminal_decision": pilot_decision["decision"],
                "intended_task": "logged-action compatibility construction pilot",
                "negative_construction": "N1 through N6",
                "known_shortcut": {
                    family: {
                        "action_summary_test_BA": report["test"][
                            "balanced_accuracy"
                        ],
                        "survives_historical_gate": report["survives"],
                    }
                    for family, report in family_decision[
                        "family_reports"
                    ].items()
                },
                "known_ambiguity": (
                    "construction labels lack alternate-world execution evidence"
                ),
                "learned_models_trained": pilot_decision[
                    "learned_baselines_trained"
                ],
                "behavioral_evidence": False,
                "permitted_role": "Groups E/R audit environments only",
                "anchors": pilot_decision["anchors"],
                "negative_candidates": dict(sorted(family_counts.items())),
            },
            {
                "branch": "rm_acv_realneg_v1",
                "terminal_decision": real_decision["decision"],
                "intended_task": "reciprocal real-action negative construction",
                "negative_construction": "N7 and N8 reciprocal swaps",
                "known_shortcut": (
                    "exact action/context balance but source metadata dependence"
                ),
                "known_ambiguity": {
                    family: {
                        "ambiguous_fraction": row["ambiguous_fraction"],
                        "distinct_mode_fraction": row["distinct_mode_fraction"],
                    }
                    for family, row in validity["families"].items()
                },
                "learned_models_trained": real_decision[
                    "learned_fair_baselines_trained"
                ],
                "behavioral_evidence": False,
                "permitted_role": "Group B diagnostics only",
                "N7_pairs": len(n7),
                "N8_pairs": len(n8),
            },
            {
                "branch": "rm_od_future_object_dynamics",
                "terminal_decision": od_decision["decision"],
                "intended_task": "future object-dynamics audit",
                "negative_construction": "cross-source/path candidate construction",
                "known_shortcut": "action/path/progress and source shortcut",
                "known_ambiguity": "not an outcome or physical validity benchmark",
                "learned_models_trained": od_decision["method_ran"],
                "behavioral_evidence": False,
                "permitted_role": "auxiliary shortcut precedent only",
            },
            {
                "branch": "rm_spd_sam3_prior_dynamics",
                "terminal_decision": spd_decision["decision"],
                "intended_task": "prior-conditioned scene-dynamics prediction",
                "negative_construction": "not applicable",
                "known_shortcut": "copy-current dominance",
                "known_ambiguity": "target not predictable under frozen task-heldout",
                "learned_models_trained": True,
                "behavioral_evidence": False,
                "permitted_role": "auxiliary task-design audit only",
            },
        ],
    }


def _roles() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = {
        "schema": "vsa-family-role-manifest-v1",
        "families": {
            family: {
                "group": group,
                "role": (
                    "easy synthetic shortcut environment"
                    if group == "E"
                    else "real-action retrieval audit environment"
                    if group == "R"
                    else "reciprocal balance diagnostic only"
                ),
                "physical_invalidity_claim": False,
                "primary_fair_training": group in {"E", "R"},
                "reciprocal_diagnostic_only": group == "B",
            }
            for family, group in FAMILIES.items()
        },
    }
    boundary = {
        "schema": "vsa-family-claim-boundary-v1",
        "construction_labels_only": True,
        "N1_to_N6": (
            "positive=logged continuation, negative=historical construction; "
            "neither is physical outcome truth"
        ),
        "N7_N8": (
            "behaviorally ambiguous reciprocal sets; usable only for "
            "action/context/source/consistency diagnostics"
        ),
        "forbidden_interpretations": [
            "negative guarantees physical failure",
            "positive guarantees task success",
            "reciprocal swap is a true counterfactual",
            "audit accuracy certifies safety",
        ],
    }
    return manifest, boundary


def execute(*, rehash_sources: bool = True) -> dict[str, Any]:
    prereg = read_json(OUT / "preregistered_config.json")
    artifacts = prereg["historical_imports"]["artifacts"]
    artifact_results = []
    for row in artifacts:
        path = Path(row["absolute_path"])
        actual_sha = sha256_file(path) if path.is_file() else None
        artifact_results.append(
            {
                **row,
                "exists": path.is_file(),
                "actual_bytes": path.stat().st_size if path.is_file() else None,
                "actual_sha256": actual_sha,
                "pass": bool(
                    path.is_file()
                    and path.stat().st_size == row["bytes"]
                    and actual_sha == row["sha256"]
                ),
            }
        )
    config_checks = [
        _verify_config_hash(branch)
        for branch in (
            "rm_acv_pilot166",
            "rm_acv_realneg_v1",
            "rm_od_future_object_dynamics",
            "rm_spd_sam3_prior_dynamics",
        )
    ]
    source_rows = _source_rows()
    source = (
        _rehash_sources(source_rows)
        if rehash_sources
        else {
            "schema": "vsa-source-content-rehash-v1",
            "pass": False,
            "skipped": True,
            "reason": "full source rehash required",
        }
    )
    import_pass = (
        all(row["pass"] for row in artifact_results)
        and all(row["pass"] for row in config_checks)
        and source["pass"]
    )
    write_json(
        OUT / "imported_artifact_manifest.json",
        {
            "schema": "vsa-imported-artifact-manifest-v1",
            "copy_performed": False,
            "artifacts": artifacts,
        },
    )
    verification = {
        "schema": "vsa-imported-hash-verification-v1",
        "artifact_checks": artifact_results,
        "historical_preregistration_checks": config_checks,
        "source_content_rehash": source,
        "pass": import_pass,
    }
    write_json(OUT / "imported_hash_verification.json", verification)
    inventory = _historical_inventory()
    inventory["import_verification_pass"] = import_pass
    write_json(OUT / "historical_branch_inventory.json", inventory)
    roles, boundary = _roles()
    write_json(OUT / "family_role_manifest.json", roles)
    write_json(OUT / "family_claim_boundary.json", boundary)
    lines = [
        "# Historical branch role table",
        "",
        "| Branch | Terminal decision | Role in VSA | Physical ground truth? |",
        "|---|---|---|---|",
    ]
    for row in inventory["branches"]:
        lines.append(
            f"| {row['branch']} | `{row['terminal_decision']}` | "
            f"{row['permitted_role']} | No |"
        )
    (OUT / "branch_role_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    append_log(
        "V1_V2_IMPORT_AND_ROLES_COMPLETE",
        imported_artifacts=len(artifact_results),
        unique_source_files=source.get("unique_source_files"),
        source_bytes=source.get("declared_bytes"),
        pass_=import_pass,
        terminal_if_failed=(
            None if import_pass else "VSA_HISTORICAL_ARTIFACTS_INCOMPLETE"
        ),
    )
    return {
        "pass": import_pass,
        "artifacts": len(artifact_results),
        "source_files": source.get("unique_source_files"),
        "source_GiB": (
            source.get("declared_bytes", 0) / 2**30
            if source.get("declared_bytes") is not None
            else None
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-source-rehash",
        action="store_true",
        help="Diagnostic only; cannot pass V1.",
    )
    args = parser.parse_args()
    result = execute(rehash_sources=not args.skip_source_rehash)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)
