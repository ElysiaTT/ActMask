#!/usr/bin/env python3
"""Rebuild the Milestone 3S paper package from frozen non-visual artifacts.

This verifier intentionally uses only the Python standard library, NumPy, and
Matplotlib.  It does not import ActMask evaluators and never writes beneath a
frozen milestone directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "outputs/actmask/milestone3s_paper_package"

SOURCES = {
    "3p_original": "outputs/actmask/milestone3p_cpu_shortcut_audit/original_distribution_results.json",
    "3p_matched": "outputs/actmask/milestone3p_cpu_shortcut_audit/matched_counterfactual_results.json",
    "3p_temporal": "outputs/actmask/milestone3p_cpu_shortcut_audit/temporal_corruption_results.json",
    "3p_final": "outputs/actmask/milestone3p_cpu_shortcut_audit/cpu_final_verification.json",
    "3q_final": "outputs/actmask/milestone3q_signed_dynamics/full_four_task/final_gate.json",
    "3q_baseline": "outputs/actmask/milestone3q_signed_dynamics/full_four_task/signed_baseline_five_seed.json",
    "3q_generation": "outputs/actmask/milestone3q_signed_dynamics/full_four_task/signed_generation_summary.json",
    "3r_completion": "outputs/actmask/milestone3r_robust_signed_dynamics/evaluation/completion_audit.json",
    "3r_selected": "outputs/actmask/milestone3r_robust_signed_dynamics/evaluation/selected_vs_analytic.json",
    "3r_controls": "outputs/actmask/milestone3r_robust_signed_dynamics/evaluation/causal_controls_five_seed.json",
    "3r_corruption": "outputs/actmask/milestone3r_robust_signed_dynamics/evaluation/learned_corruption_matrix.json",
    "v1_config": "outputs/actmask/milestone3r_nl_v1/preregistered_config.json",
    "v1_evaluation": "outputs/actmask/milestone3r_nl_v1/evaluation/full_evaluation.json",
    "v1_fairness": "outputs/actmask/milestone3r_nl_v1/fairness_fixes.json",
    "v2_probe_config": "outputs/actmask/milestone3r_nl_v2/preregistered_config.json",
    "v2_probe_decision": "outputs/actmask/milestone3r_nl_v2/probe_decision.json",
    "v2_probe_evaluation": "outputs/actmask/milestone3r_nl_v2/evaluation/probe_evaluation.json",
    "v2_forensic": "outputs/actmask/milestone3r_nl_v2/v1_forensic_audit.json",
    "v2_candidate_audit": "outputs/actmask/milestone3r_nl_v2/evaluation/candidate_diversity_audit.json",
    "v2_ood_overlap": "outputs/actmask/milestone3r_nl_v2/evaluation/ood_overlap_audit.json",
    "v2_full_config": "outputs/actmask/milestone3r_nl_v2/full_run_config.json",
    "v2_full_decision": "outputs/actmask/milestone3r_nl_v2/full_decision.json",
    "v2_full_evaluation": "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json",
}
FULL_REPORTS = {
    "id_core": "outputs/actmask/milestone3r_nl_v2/full_raw/id_core/generation_report.json",
    "id_rotation": "outputs/actmask/milestone3r_nl_v2/full_raw/id_rotation/generation_report.json",
    "physical_parameter_ood": "outputs/actmask/milestone3r_nl_v2/full_raw/physical_parameter_ood/generation_report.json",
    "temporal_delay_ood": "outputs/actmask/milestone3r_nl_v2/full_raw/temporal_delay_ood/generation_report.json",
    "held_mechanism_ood": "outputs/actmask/milestone3r_nl_v2/full_raw/held_mechanism_ood/generation_report.json",
}
EXPECTED_HASHES = {
    "v1_config": "79d04d3204af01386d1a178a874d537a345bd2a8ca94545e750fcb4d50209b9f",
    "v2_probe_config": "9dc510bdec94e4715df96f6cc6f960f932d4dd37babea957e0f14325cf98e564",
    "v2_full_config": "12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47",
}


def read_json(relative: str):
    with (ROOT / relative).open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def status(name, passed, expected, observed, source, note=""):
    return {"name": name, "status": "PASS" if passed else "FAIL", "expected": expected,
            "observed": observed, "source": source, "note": note}


def close(a, b, tolerance=1e-12):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def get_method(records, method, split="test"):
    values = [r["pair_order_accuracy"] for r in records if r["method"] == method and r["split"] == split]
    if not values:
        raise KeyError(f"No {method!r} records for {split}")
    return mean(values)


def tex(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_table(out: Path, name: str, rows: list[dict], fields: list[str]) -> None:
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    dump(table_dir / f"{name}.json", {"columns": fields, "rows": rows})
    aligns = "l" * len(fields)
    lines = [f"\\begin{{tabular}}{{{aligns}}}", "\\toprule", " & ".join(tex(x) for x in fields) + r" \\", "\\midrule"]
    lines += [" & ".join(tex(row.get(field, "")) for field in fields) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (table_dir / f"{name}.tex").write_text("\n".join(lines) + "\n")


def runtime_info():
    info = {"active_python": sys.executable, "conda_environment_path": sys.prefix,
            "python_version": sys.version, "platform": platform.platform()}
    try:
        import torch
        info.update({"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                     "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as exc: info["torch_error"] = repr(exc)
    for module, label in (("mani_skill", "maniskill_version"), ("sapien", "sapien_version")):
        try:
            info[label] = __import__(module).__version__
        except Exception as exc: info[label] = f"unavailable: {exc!r}"
    repaired = []
    site = Path(sys.prefix) / "lib/python3.11/site-packages"
    if site.exists():
        for path in sorted(site.iterdir()):
            if path.is_symlink() and "/home/tzh/conda_envs/papers" in os.readlink(path):
                repaired.append({"path": str(path), "target": os.readlink(path), "resolved": str(path.resolve())})
    info["repaired_symlink_targets"] = repaired
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        info["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except Exception:
        info["git_commit"] = None; info["git_status"] = ["Repository metadata unavailable in this workspace."]
    return info


def artifact_manifest(data, out):
    entries, failures = [], []
    for name, relative in {**SOURCES, **FULL_REPORTS}.items():
        path = ROOT / relative
        exists = path.is_file()
        entry = {"name": name, "path": relative, "exists": exists}
        if exists:
            entry.update({"sha256": sha256(path), "size_bytes": path.stat().st_size})
        else: failures.append(name)
        entries.append(entry)
    by_name = {x["name"]: x for x in entries}
    hash_checks = []
    for name, expected in EXPECTED_HASHES.items():
        observed = by_name[name].get("sha256")
        hash_checks.append(status(f"{name}_sha256", observed == expected, expected, observed, SOURCES[name]))
    # Frozen reports must agree with the full config's execution budget and decision config hash.
    if not failures:
        full_config, full_eval, full_decision = data["v2_full_config"], data["v2_full_evaluation"], data["v2_full_decision"]
        config_hash = by_name["v2_full_config"]["sha256"]
        hash_checks += [
            status("full_evaluation_config_binding", full_eval["full_config_sha256"] == config_hash, config_hash, full_eval["full_config_sha256"], SOURCES["v2_full_evaluation"]),
            status("full_decision_config_binding", full_decision["full_run_config_sha256"] == config_hash, config_hash, full_decision["full_run_config_sha256"], SOURCES["v2_full_decision"]),
            status("decision_execution_budget_binding", full_decision["candidate_executions"] == full_config["candidate_execution_budget"], full_config["candidate_execution_budget"], full_decision["candidate_executions"], SOURCES["v2_full_decision"]),
        ]
    manifest = {"schema": "milestone3s-artifact-manifest-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime_info(), "artifacts": entries, "checks": hash_checks,
                "passed": not failures and all(x["status"] == "PASS" for x in hash_checks),
                "missing_artifacts": failures,
                "freeze_policy": "Source artifacts were read only. New outputs are confined to milestone3s_paper_package."}
    dump(out / "artifact_manifest.json", manifest)
    return manifest


def independent_audit(data, out):
    checks = []
    # 3P: direct independent extraction of frozen aggregate numbers.
    original, matched, temporal = data["3p_original"], data["3p_matched"], data["3p_temporal"]
    def original_pair(method):
        seeds = original["methods"][method]["seed_results"]
        return mean(s["metrics"]["pairwise_ranking_accuracy"] for s in seeds)
    static_original = original_pair("CurrentPositionProximity")
    temporal_original = original_pair("TimeAlignedRelativeFeatures")
    matched_static = matched["counterfactual_pair_accuracy"]["CurrentPositionProximity"]["pair_accuracy"]
    matched_temporal = matched["counterfactual_pair_accuracy"]["TimeAlignedRelativeFeatures"]["pair_accuracy"]
    temporal_correct = temporal["correct"]["counterfactual_pair_accuracy"]["pair_accuracy"]
    for name, val, expected, src in [
        ("3p_static_original", static_original, 1.0, "3p_original"), ("3p_temporal_original", temporal_original, 0.9583333333333334, "3p_original"),
        ("3p_matched_static", matched_static, 0.5, "3p_matched"), ("3p_matched_temporal", matched_temporal, 0.9166666666666666, "3p_matched"),
        ("3p_temporal_correct", temporal_correct, 0.9166666666666666, "3p_temporal")]:
        checks.append(status(name, close(val, expected), expected, val, SOURCES[src]))
    checks.append(status("3p_shortcut_conclusion", static_original > .9 and matched_static == .5 and matched_temporal > matched_static,
                         "original static high; matched static chance; temporal above static", {"original_static": static_original, "matched_static": matched_static, "matched_temporal": matched_temporal}, SOURCES["3p_matched"]))
    # 3Q: final gate is a compact, frozen aggregate; independently verify all stated conditions.
    q = data["3q_final"]; qacc = q["five_seed_pair_order_accuracy"]
    for method, expected in {"action_mlp": .5, "static_mlp": .5, "unordered_history_mlp": .5, "sorted_history_mlp": .5, "signed_finite_difference": 1., "ordered_temporal_mlp": 1.}.items():
        checks.append(status(f"3q_{method}", qacc[method] == expected, expected, qacc[method], SOURCES["3q_final"]))
    checks += [status("3q_exact_reversal_drop", all(v == 0. for v in q["exact_reversed_history"].values()), 0., q["exact_reversed_history"], SOURCES["3q_final"]),
               status("3q_four_task_support", q["four_task_generation"]["tasks"] == 4, 4, q["four_task_generation"]["tasks"], SOURCES["3q_final"]),
               status("3q_ood_phase_delay", q["phase_delay_mechanism_ood"]["signed_finite_difference"] == 1. and q["phase_delay_mechanism_ood"]["ordered_temporal_mlp"] == 1., 1., q["phase_delay_mechanism_ood"], SOURCES["3q_final"])]
    # 3R robust intermediate milestone.
    r, rs = data["3r_completion"], data["3r_selected"]
    pair = r["robust_pair_order"]
    checks += [status("3r_gru_pair_order", pair["learned"] == .9057, .9057, pair["learned"], SOURCES["3r_completion"]),
               status("3r_lasttwo_pair_order", pair["analytic"] == .54153125, .54153125, pair["analytic"], SOURCES["3r_completion"]),
               status("3r_learned_minus_analytic", close(pair["delta"], pair["learned"] - pair["analytic"]), pair["learned"] - pair["analytic"], pair["delta"], SOURCES["3r_completion"]),
               status("3r_human_decision_reason", r["stage_decision"] == "E_human_decision_required" and not r["checks"]["two_nonlinear_ood_positive_axes"], "human decision; nonlinear OOD gate not met", {"decision": r["stage_decision"], "ood": r["checks"]["two_nonlinear_ood_positive_axes"]}, SOURCES["3r_completion"]),
               status("3r_bootstrap_groups", rs["grouped_bootstrap"]["groups"] == 32, 32, rs["grouped_bootstrap"]["groups"], SOURCES["3r_selected"])]
    # v1 constraints.
    v1 = data["v1_evaluation"]
    checks += [status("v1_learned_advantage", v1["learned_over_analytic_grouped_bootstrap"]["delta"] == .058, .058, v1["learned_over_analytic_grouped_bootstrap"]["delta"], SOURCES["v1_evaluation"]),
               status("v1_final_two_control", v1["causal_diagnostics"]["earlier_history_advantage"] == .5, .5, v1["causal_diagnostics"]["earlier_history_advantage"], SOURCES["v1_evaluation"]),
               status("v1_generic_reversal_failure", v1["causal_diagnostics"]["reversal_causal_drop"] == 0., 0., v1["causal_diagnostics"]["reversal_causal_drop"], SOURCES["v1_evaluation"]),
               status("v1_candidate_limitation", "same branch outcome" in v1["candidate_action_limitation"], "candidate degeneracy documented", v1["candidate_action_limitation"], SOURCES["v1_evaluation"]),
               status("v1_no_nonlinear_ood_claim", "no nonlinear-OOD pass claim" in v1["ood_status"], "explicit no-pass limitation", v1["ood_status"], SOURCES["v1_evaluation"])]
    # v2 probe.
    probe, forensic, candidate, overlap = data["v2_probe_evaluation"], data["v2_forensic"], data["v2_candidate_audit"], data["v2_ood_overlap"]
    probe_gru, probe_analytic = get_method(probe["records"], "GRU"), get_method(probe["records"], "MultiFrameLinearVelocity")
    checks += [status("v2_probe_valid_pair_swap", "maps each v1 pair exactly" in forensic["findings"]["valid_v2_operator_candidate"], "exact pair mapping", forensic["findings"]["valid_v2_operator_candidate"], SOURCES["v2_forensic"]),
               status("v2_probe_nondegenerate_fraction", candidate["summary"]["nondegenerate_world_fraction"] >= .8, ">=0.8", candidate["summary"]["nondegenerate_world_fraction"], SOURCES["v2_candidate_audit"]),
               status("v2_probe_branch_change", candidate["summary"]["mean_branch_outcome_change_rate"] >= .25, ">=0.25", candidate["summary"]["mean_branch_outcome_change_rate"], SOURCES["v2_candidate_audit"]),
               status("v2_probe_learned_advantage", probe_gru - probe_analytic >= .03, ">=0.03", probe_gru - probe_analytic, SOURCES["v2_probe_evaluation"]),
               status("v2_probe_ood_disjoint", not any(v for group in overlap.values() for key, v in group.items() if key.endswith("overlap")), False, overlap, SOURCES["v2_ood_overlap"])]
    # v2 full.  The execution count is recomputed from independent raw-generation reports.
    full, config, decision = data["v2_full_evaluation"], data["v2_full_config"], data["v2_full_decision"]
    executions = sum(data[f"report_{name}"]["candidate_executions"] for name in FULL_REPORTS)
    gru, analytic = get_method(full["records"], "GRUFiveSeed"), get_method(full["records"], "MultiFrameLinearVelocity")
    checks += [status("v2_full_execution_count", executions == 92000 == config["candidate_execution_budget"] == decision["candidate_executions"], 92000, executions, " + ".join(FULL_REPORTS.values())),
               status("v2_full_learned_minus_analytic", close(gru - analytic, full["bootstrap"]["delta"]) and close(full["bootstrap"]["delta"], .4), .4, gru - analytic, SOURCES["v2_full_evaluation"]),
               status("v2_full_grouped_ci", full["bootstrap"]["ci95"] == [0.39508333333333334, 0.4051666666666667], [0.39508333333333334, 0.4051666666666667], full["bootstrap"]["ci95"], SOURCES["v2_full_evaluation"]),
               status("v2_full_ood_axes", all(row["learned"] > row["analytic"] for row in full["ood"]), "learned > analytic on all 3 OOD axes", full["ood"], SOURCES["v2_full_evaluation"]),
               status("v2_full_rankings", all(full["rankings"][c]["dynamic"]["top1_success"] == 1. and full["rankings"][c]["dynamic"]["corrected_utility_gain_ndcg"] == 1. for c in ("C5","C10","C20")), "C5/C10/C20 dynamic Top-1,NDCG=1", full["rankings"], SOURCES["v2_full_evaluation"]),
               status("v2_full_c20_latency", full["c20_latency_ms"]["p95"] <= .816, "<=0.816 ms", full["c20_latency_ms"]["p95"], SOURCES["v2_full_evaluation"]),
               status("v2_full_decision", decision["decision"] == "NONLINEAR_OOD_GATE_PASSED" and full["full_gate_passed"], "NONLINEAR_OOD_GATE_PASSED", {"decision": decision["decision"], "gate": full["full_gate_passed"]}, SOURCES["v2_full_decision"])]
    audit = {"schema": "milestone3s-independent-metric-audit-v1", "method": "Independent JSON arithmetic and consistency checks; no ActMask evaluator imported.", "checks": checks,
             "passed": all(c["status"] == "PASS" for c in checks)}
    dump(out / "independent_metric_audit.json", audit)
    return audit


def red_team(data, out):
    import numpy as np
    checks = []
    cfg, full, decision = data["v2_probe_config"], data["v2_full_evaluation"], data["v2_full_decision"]
    npz_paths = list((ROOT / "outputs/actmask/milestone3r_nl_v2/full").glob("*/model_inputs.npz"))
    forbidden = set(cfg["forbidden_fair_fields"])
    input_keys = {str(p.relative_to(ROOT)): sorted(np.load(p, allow_pickle=False).files) for p in npz_paths}
    found = sorted({key for keys in input_keys.values() for key in keys if key in forbidden or "object" in key.lower() or key in {"branch", "family", "mechanism"}})
    checks.append({"issue": "hidden_field_and_object_id_leakage", "status": "PASS" if not found else "FAIL", "evidence": {"forbidden": sorted(forbidden), "model_input_keys": input_keys, "forbidden_keys_found": found}, "affected_claim": "fair state-only verifier comparison", "recommended_fix": "Remove forbidden columns and regenerate only under a new protocol if any are found.", "blocks_paper_submission": bool(found)})
    diversity = full["candidate_diversity"]
    checks += [
      {"issue":"candidate_template_or_index_leakage", "status":"PASS" if diversity["candidate_template_rate_span"] == 0 and diversity["max_candidate_index_leakage"] == 0 else "FAIL", "evidence":diversity, "affected_claim":"candidate ranking is not template/index lookup", "recommended_fix":"Rebalance/re-generate templates under a new protocol.", "blocks_paper_submission": diversity["candidate_template_rate_span"] != 0 or diversity["max_candidate_index_leakage"] != 0},
      {"issue":"degenerate_candidate_worlds", "status":"PASS" if diversity["nondegenerate_world_fraction"] == 1 and diversity["degenerate_world_rate"] == 0 else "FAIL", "evidence":diversity, "affected_claim":"action-diverse ranking evidence", "recommended_fix":"Exclude/re-generate degenerate worlds under preregistered criteria.", "blocks_paper_submission": diversity["nondegenerate_world_fraction"] != 1},
      {"issue":"train_ood_range_overlap", "status":"PASS" if cfg["id_ranges"]["damping"][1] < cfg["ood_axes"]["physical_parameter"]["test"]["damping"][0] and cfg["id_ranges"]["execution_delay_steps"][1] < cfg["ood_axes"]["temporal_delay"]["test"]["execution_delay_steps"][0] else "FAIL", "evidence":{"id_ranges":cfg["id_ranges"],"ood_axes":cfg["ood_axes"]}, "affected_claim":"OOD generalization", "recommended_fix":"Use disjoint preregistered ranges.", "blocks_paper_submission":False},
      {"issue":"success_definition_coupling", "status":"PASS" if not cfg["success"]["branch_specific_thresholds"] and "GPU-PhysX" in cfg["success"]["criterion"] and "common 0.055 radius" in cfg["success"]["criterion"] else "FAIL", "evidence":cfg["success"], "affected_claim":"fair comparison to analytic baselines", "recommended_fix":"Use one simulator-only threshold for all methods.", "blocks_paper_submission":True},
      {"issue":"model_selection_on_test_data", "status":"PASS" if full["selected_learned"] == max(full["validation"], key=full["validation"].get) else "FAIL", "evidence":{"selected":full["selected_learned"],"validation":full["validation"]}, "affected_claim":"learned baseline selection", "recommended_fix":"Select only on validation and keep test sealed.", "blocks_paper_submission":False},
      {"issue":"confidence_interval_misuse", "status":"PASS" if full["bootstrap"]["groups"] == 6000 and full["bootstrap"]["ci95"][0] <= full["bootstrap"]["delta"] <= full["bootstrap"]["ci95"][1] else "FAIL", "evidence":full["bootstrap"], "affected_claim":"learned-over-analytic advantage", "recommended_fix":"Persist resamples or recompute grouped bootstrap in a new audit.", "blocks_paper_submission":False},
      {"issue":"posthoc_threshold_changes", "status":"WARNING", "evidence":{"full_gate":data["v2_full_config"]["full_gate"],"data_audit_refresh":full["data_audit_refresh"]}, "affected_claim":"strict preregistration interpretation", "recommended_fix":"Describe the C20 audit bookkeeping correction and retain the frozen configuration hash; do not present it as a new experimental threshold.", "blocks_paper_submission":False},
      {"issue":"stale_v1_as_v2_proof", "status":"PASS" if decision["full_run_config_sha256"] == full["full_config_sha256"] and "v2" in data["v2_full_config"]["extension"] else "FAIL", "evidence":{"decision_config":decision["full_run_config_sha256"],"eval_config":full["full_config_sha256"],"extension":data["v2_full_config"]["extension"]}, "affected_claim":"v2 nonlinear OOD pass", "recommended_fix":"Cite only v2 full config/evaluation/decision for the final claim.", "blocks_paper_submission":False}
    ]
    # Split and label balance from persisted full ID labels, not evaluator code.
    labels_paths = [ROOT / "outputs/actmask/milestone3r_nl_v2/full/id/labels.npz"]
    label_values, split_keys = [], []
    for path in labels_paths:
        archive = np.load(path, allow_pickle=False)
        for key in archive.files:
            if key in {"labels", "success", "branch"}: label_values.extend(np.asarray(archive[key]).reshape(-1).tolist())
            if "split" in key.lower(): split_keys.append(key)
    binary = [x for x in label_values if x in (0, 1, False, True)]
    balance = mean(binary) if binary else None
    checks.append({"issue":"label_imbalance_artifact", "status":"PASS" if balance == .5 else "WARNING", "evidence":{"label_files": [str(p.relative_to(ROOT)) for p in labels_paths],"binary_label_fraction":balance,"split_keys":split_keys}, "affected_claim":"pair-order chance control", "recommended_fix":"If labels are not balanced, report a balanced metric; current frozen pair construction is balanced.", "blocks_paper_submission":False})
    metadata_path = ROOT / "outputs/actmask/milestone3r_nl_v2/full/id/metadata.jsonl"
    world_splits = {}
    with metadata_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            world = (row["condition"], row["family"], row["mechanism"], row["world_id"])
            world_splits.setdefault(world, set()).add(row["split"])
    contaminated = [list(key) for key, splits in world_splits.items() if len(splits) != 1]
    checks.append({"issue":"grouped_split_leakage", "status":"PASS" if not contaminated else "FAIL", "evidence":{"metadata":str(metadata_path.relative_to(ROOT)),"worlds_checked":len(world_splits),"contaminated_worlds":contaminated[:10]}, "affected_claim":"held-out test validity", "recommended_fix":"Split at the world level before model fitting.", "blocks_paper_submission":bool(contaminated)})
    audit = {"schema":"milestone3s-red-team-audit-v1", "scope":"Attempts to invalidate the state-only v2 claim using frozen config, inputs, labels and evaluation.", "checks":checks,
             "blocking_fail_count":sum(x["status"] == "FAIL" and x["blocks_paper_submission"] for x in checks), "warning_count":sum(x["status"] == "WARNING" for x in checks)}
    audit["passed"] = audit["blocking_fail_count"] == 0
    dump(out / "red_team_audit.json", audit)
    return audit


def make_tables(data, out):
    full = data["v2_full_evaluation"]
    progression = [
      {"milestone":"3P","goal":"Expose static shortcut","shortcut_tested":"static geometry/action correlation","strongest_control":"matched static: 0.500","learned_method":"TimeAlignedRelativeFeatures","key_result":"matched temporal: 0.917","decision":"shortcut demonstrated","limitation":"CPU procedural audit"},
      {"milestone":"3Q","goal":"Force signed temporal order","shortcut_tested":"unordered/sorted history","strongest_control":"static/action/unordered: 0.500","learned_method":"ordered temporal MLP","key_result":"signed methods: 1.000","decision":"signed-order pass","limitation":"linear signed dynamics"},
      {"milestone":"3R","goal":"Robustness under corruptions","shortcut_tested":"corruption/ambiguity","strongest_control":"LastTwo: 0.542","learned_method":"GRU","key_result":"GRU−analytic: +0.364","decision":"human decision required","limitation":"nonlinear OOD gate not met"},
      {"milestone":"3R-NL-v1","goal":"Nonlinear state-only test","shortcut_tested":"final-two/history reversal","strongest_control":"RobustLinearVelocity","learned_method":"GRU","key_result":"GRU−analytic: +0.058","decision":"no nonlinear-OOD pass","limitation":"invalid reversal/candidate degeneracy"},
      {"milestone":"3R-NL-v2","goal":"Nonlinear OOD with matched final two","shortcut_tested":"early pair-swap/action diversity","strongest_control":"MultiFrameLinearVelocity: 0.600","learned_method":"GRU","key_result":"GRU−analytic: +0.400; 3 OOD axes positive","decision":"full gate passed","limitation":"state-only, simulated"},]
    write_table(out,"milestone_progression",progression,list(progression[0]))
    records = full["records"]
    def f(method): return f"{get_method(records, method):.3f}"
    ood = {x["axis"]:x for x in full["ood"]}
    c20 = full["rankings"]["C20"]
    main = []
    mapping=[("Static","StaticMLP"),("Action-only","ActionMLP"),("Unordered history","UnorderedHistoryMLP"),("LastTwo","LastTwoFrameFiniteDifference"),("MultiFrameLinearVelocity","MultiFrameLinearVelocity"),("Strongest fair analytic","MultiFrameLinearVelocity"),("GRU","GRUFiveSeed"),("TemporalConv1D","TemporalConv1D")]
    for label, method in mapping:
        dynamic = label == "GRU"
        ranking = c20["dynamic"] if dynamic else (c20.get(method) or c20.get("StaticMLP") if label == "Static" else c20.get("ActionMLP") if label == "Action-only" else None)
        main.append({"method":label,"ID_pair_order":f(method),"physical_OOD":f"{ood['physical_parameter_ood']['learned']:.3f}" if dynamic else (f"{ood['physical_parameter_ood']['analytic']:.3f}" if method == "MultiFrameLinearVelocity" else "not measured"),"temporal_OOD":f"{ood['temporal_delay_ood']['learned']:.3f}" if dynamic else (f"{ood['temporal_delay_ood']['analytic']:.3f}" if method == "MultiFrameLinearVelocity" else "not measured"),"held_mechanism_OOD":f"{ood['held_mechanism_ood']['learned']:.3f}" if dynamic else (f"{ood['held_mechanism_ood']['analytic']:.3f}" if method == "MultiFrameLinearVelocity" else "not measured"),"C20_Top1":f"{ranking['top1_success']:.3f}" if ranking else "not measured","NDCG":f"{ranking['corrected_utility_gain_ndcg']:.3f}" if ranking else "not measured","regret":f"{ranking['normalized_regret']:.3f}" if ranking else "not measured","latency_ms":f"{full['c20_latency_ms']['p95']:.3f} p95" if dynamic else "not measured"})
    write_table(out,"main_v2_results",main,list(main[0]))
    interventions=full["interventions"]
    causal=[
      {"condition":"correct history","pair_order":f"{interventions['correct']:.3f}","swap_accuracy":f"{full['swap']['counterfactual_swap_accuracy']:.3f}","score_swap_consistency":f"{full['swap']['score_swap_consistency']:.3f}","interpretation":"valid paired history"},
      {"condition":"pair-swap","pair_order":"not separately scored","swap_accuracy":f"{full['swap']['counterfactual_swap_accuracy']:.3f}","score_swap_consistency":f"{full['swap']['score_swap_consistency']:.3f}","interpretation":"maps paired alternative exactly"},
      {"condition":"final-two-only","pair_order":f"{interventions['last_two_only']:.3f}","swap_accuracy":"not measured","score_swap_consistency":"not measured","interpretation":"chance; earlier-history requirement"},
      {"condition":"earlier-zeroed","pair_order":"not measured","swap_accuracy":"not measured","score_swap_consistency":"not measured","interpretation":"not evaluated in frozen full run"},
      {"condition":"unordered","pair_order":f"{get_method(full['records'],'UnorderedHistoryMLP'):.3f}","swap_accuracy":"not measured","score_swap_consistency":"not measured","interpretation":"order-invariant control at chance"},
      {"condition":"mismatched history","pair_order":f"{interventions['early_independent_permutation']:.3f}","swap_accuracy":"not measured","score_swap_consistency":"not measured","interpretation":"independent early-frame permutation"},
      {"condition":"generic full reversal diagnostic","pair_order":f"{interventions['generic_full_reverse']:.3f}","swap_accuracy":"not measured","score_swap_consistency":"not measured","interpretation":"invalid diagnostic; not a valid pair swap"},]
    write_table(out,"causal_controls",causal,list(causal[0]))
    d=full["candidate_diversity"]
    div=[{"nondegenerate_world_fraction":d["nondegenerate_world_fraction"],"success_failure_counts":"each template success rate 0.500; 10 positive/10 negative trajectories","branch_dependent_outcome_change_rate":d["mean_branch_outcome_change_rate"],"template_leakage_rate_span":d["candidate_template_rate_span"],"index_leakage_max":d["max_candidate_index_leakage"]}]
    write_table(out,"candidate_diversity",div,list(div[0]))


def figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    full=data["v2_full_evaluation"]
    provenance={}
    def save(name, sources):
        plt.tight_layout(); plt.savefig(figdir/f"{name}.png",dpi=180); plt.savefig(figdir/f"{name}.svg"); plt.close(); provenance[name]={"sources":sources,"generated_files":[f"{name}.png",f"{name}.svg"]}
    # diagrams, deliberately data-free conceptual rendering with source citation.
    def flow(name, labels, sources):
        plt.figure(figsize=(12,2.2)); ax=plt.gca(); ax.set_axis_off()
        xs=[.04+i*.92/(len(labels)-1) for i in range(len(labels))]
        for i,(x,label) in enumerate(zip(xs,labels)):
            ax.text(x,.52,label,ha='center',va='center',wrap=True,fontsize=9,bbox=dict(boxstyle='round,pad=.45',fc='#eaf2f8',ec='#2874a6'))
            if i: ax.annotate('',xy=(x-.07,.52),xytext=(xs[i-1]+.07,.52),arrowprops={'arrowstyle':'->','lw':1.5})
        save(name,sources)
    flow('shortcut_removal_layers',["ordinary data", "static shortcut", "matched static\ncounterfactual", "order-invariant\nmotion shortcut", "signed-order\ncounterfactual", "nonlinear OOD +\nfinal-two matching"],[SOURCES['3p_matched'],SOURCES['3q_final'],SOURCES['v2_full_config']])
    flow('counterfactual_pair_construction',["same final two frames", "branch-identifying early frames", "T(h)=reverse(h[:4])+h[4:]", "same candidate set", "opposite simulator outcome"],[SOURCES['v2_probe_config'],SOURCES['v2_forensic']])
    flow('v1_failure_v2_repair',["v1 generic reversal\nnot paired alternative", "v1 C20 candidate\ndegeneracy", "v2 exact early\npair-swap", "v2 action-diverse\ncandidates"],[SOURCES['v1_evaluation'],SOURCES['v2_forensic'],SOURCES['v2_full_evaluation']])
    # performance chart
    rows=[('Static',get_method(full['records'],'StaticMLP')),('Action',get_method(full['records'],'ActionMLP')),('Unordered',get_method(full['records'],'UnorderedHistoryMLP')),('Analytic',get_method(full['records'],'MultiFrameLinearVelocity')),('GRU',get_method(full['records'],'GRUFiveSeed'))]
    names=[x[0] for x in rows]; ids=[x[1] for x in rows]; ood=[None,None,None,mean(x['analytic'] for x in full['ood']),mean(x['learned'] for x in full['ood'])]
    plt.figure(figsize=(8,4)); x=list(range(len(names))); plt.bar([i-.18 for i in x],ids,.36,label='ID pair-order'); plt.bar([i+.18 for i in x],[v or 0 for v in ood],.36,label='mean OOD pair-order');
    for i,v in enumerate(ood):
        if v is None: plt.text(i+.18,.03,'not\nmeasured',ha='center',fontsize=7,rotation=90)
    plt.xticks(x,names);plt.ylim(0,1.1);plt.ylabel('accuracy');plt.legend();save('main_performance',[SOURCES['v2_full_evaluation']])
    rates=full['candidate_diversity']['candidate_template_success_rates'];plt.figure(figsize=(9,3.4));plt.bar(list(map(int,rates)),list(rates.values()));plt.axhline(.5,color='black',ls='--');plt.xlabel('candidate template index');plt.ylabel('success rate');plt.ylim(0,1);save('candidate_diversity_histogram',[SOURCES['v2_full_evaluation']])
    plt.figure(figsize=(6,3.6));plt.bar(['C20 mean','C20 p95','GPU peak / 1000'],[full['c20_latency_ms']['mean'],full['c20_latency_ms']['p95'],full['gpu_peak_allocated_mb']/1000],color=['#5dade2','#2e86c1','#85c1e9']);plt.ylabel('ms (first two); GB proxy (third)');save('latency_resource_summary',[SOURCES['v2_full_evaluation']])
    dump(figdir/'figure_provenance.json',provenance)


def load():
    data={name:read_json(path) for name,path in SOURCES.items()}
    data.update({f"report_{name}":read_json(path) for name,path in FULL_REPORTS.items()})
    return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT_DEFAULT);ap.add_argument('--no-figures',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    data=load(); manifest=artifact_manifest(data,out)
    if not manifest['passed']:
        raise SystemExit('Required frozen artifact missing or inconsistent; see artifact_manifest.json')
    metric=independent_audit(data,out)
    if not metric['passed']:
        raise SystemExit('Headline metric discrepancy; see independent_metric_audit.json')
    red=red_team(data,out)
    if not red['passed']:
        raise SystemExit('Blocking red-team finding; see red_team_audit.json')
    make_tables(data,out)
    if not args.no_figures: figures(data,out)
    dump(out/'milestone3s_decision.json',{'decision':'A. PAPER PACKAGE READY','manifest_passed':manifest['passed'],'independent_metric_audit_passed':metric['passed'],'red_team_blocking_fail_count':red['blocking_fail_count'],'red_team_warning_count':red['warning_count'],'visual_stage_authorized':False,'rationale':'Frozen state-only evidence is internally bound, independently checked, and package artifacts were generated. Visual execution remains unauthorized.'})
    print(f'Paper package generated at {out}')

if __name__ == '__main__': main()
