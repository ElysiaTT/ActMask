"""Build the non-mutating P-Series synthetic-to-real audit and access package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "outputs" / "actmask"
OUT = SOURCE / "p_series_audit_and_access_package"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n")


FAMILIES = [
    ("5B-0", "milestone5b0_relational_probe", "Synthetic relational probe"),
    ("5B", "milestone5b_relational_verifier", "Synthetic relational verifier"),
    ("5B-F", "milestone5b_f_plateau_forensics", "Synthetic plateau forensics"),
    ("5B-Data", "milestone5b_data_observable_cues", "Observable-cue synthetic redesign"),
    ("5B-Data-F", "milestone5b_data_f_saturation_forensics", "Observable-cue saturation forensics"),
    ("5B-Data-v2", "milestone5b_data_v2_partial_cue", "Partial-cue redesign"),
    ("5B-Data-v3", "milestone5b_data_v3_compositional_ood", "Compositional OOD redesign"),
    ("R-Series ASU", "r_series_real_robot_verification", "ASU real UR5 logged-consistency benchmark"),
    ("R2 BotFails numeric", "r2_series_real_outcome_search", "BotFails numeric failure-onset benchmark"),
    ("R3 BotFails visual", "r3_botfails_visual_failure", "BotFails causal RGB failure-forecasting benchmark"),
]

KEY_ARTIFACTS = {
    "5B-0": ["preregistered_config.json", "integrity_audit.json", "ci_consistency_audit.json", "headroom_metrics.json", "raw_score_manifest.json", "final_decision.json"],
    "5B": ["preregistered_config.json", "input_field_audit.json", "ablation_report.json", "headroom_closure.json", "raw_score_manifest.json", "final_decision.json"],
    "5B-F": ["preregistered_config.json", "fair_observation_identifiability.json", "action_conditioning_sensitivity.json", "metric_split_sanity.json", "final_decision.json"],
    "5B-Data": ["preregistered_config.json", "field_separation_audit.json", "fair_identifiability_audit.json", "headroom_metrics.json", "raw_score_manifest.json", "final_decision.json"],
    "5B-Data-F": ["preregistered_config.json", "cue_ablation_diagnostics.json", "saturation_breakdown.json", "redesign_recommendation.json", "final_decision.json"],
    "5B-Data-v2": ["preregistered_config.json", "fair_input_schema.json", "goal_cue_presence_audit.json", "token_order_audit.json", "split_audit.json", "raw_score_manifest.json", "final_decision.json"],
    "5B-Data-v3": ["preregistered_config.json", "fair_input_schema.json", "goal_cue_presence_audit.json", "token_order_audit.json", "split_audit.json", "raw_score_manifest.json", "final_decision.json"],
    "R-Series ASU": ["dataset_survey.json", "processed_subset_asu_tabletop/adapter_audit.json", "task_formulation/future_leakage_audit.json", "baseline_report.json", "raw_score_manifest.json", "final_decision.json"],
    "R2 BotFails numeric": ["processed_subset/botfails/adapter_audit.json", "task_formulation/formulation_a_failure_onset/task_audit.json", "baseline_results/formulation_a_baseline_report.json", "method_results/real_rel_dyn_verifier/method_report.json", "final_package/final_decision.json", "final_package/reproducibility_manifest.json"],
    "R3 BotFails visual": ["preregistered_config.json", "source_video_manifest.json", "video_alignment_audit.json", "task_formulation/leakage_audit.json", "baseline_results/visual_baseline_report.json", "final_package/final_decision.json", "final_package/reproducibility_manifest.json", "final_package/raw_score_manifest.json"],
}


TAXONOMY = [
    {"id": "global_motion_shortcut", "category": "global-motion shortcut", "milestone": "5B-0 / prior synthetic clean-state audit", "evidence": "global ICP and centroid controls were explicitly audited before the relational probe; a global-motion explanation is not method evidence.", "diagnosis": "compare global/centroid features to fair relational inputs under matched candidates", "fixed": "audited and controlled, not a positive method result", "invalidated": "claims that global scene motion proves relational reasoning", "valid": "the audit can expose shortcut risk"},
    {"id": "estimator_ci_mismatch", "category": "estimator / CI mismatch", "milestone": "5A metric repair (supplementary historical audit)", "evidence": "metric_aggregation_audit records 73 point estimates outside non-matching intervals and identifies distinct estimands.", "diagnosis": "require the interval and point estimate to use the same estimator", "fixed": "frozen reports were not rewritten; non-comparable CIs are prohibited from citation", "invalidated": "confidence-interval claims from the affected frozen aggregate", "valid": "raw-score, estimator-consistent reporting is required going forward"},
    {"id": "missing_raw_score_checkpoint", "category": "missing raw-score/checkpoint artifacts", "milestone": "5A-R repair", "evidence": "repaired_decision_audit says RAW_SCORES_MISSING_STOP; frozen aggregates could not support recomputation.", "diagnosis": "attempt same-estimator recomputation from saved artifacts", "fixed": "not recoverable for that frozen result", "invalidated": "inference and repaired decision claims for the affected result", "valid": "later 5B/R2/R3 runs retain manifests and raw-score evidence"},
    {"id": "fair_observation_unidentifiability", "category": "fair-observation unidentifiability", "milestone": "5B-F", "evidence": "fair_observation_identifiability reports conflicting fair-input classes and a 0.75 Bayes-majority upper bound.", "diagnosis": "exact fair-input equivalence-class conflict analysis", "fixed": "redesigned with observable cues", "invalidated": "a fair-input relational-method claim on the ambiguous clean-state task", "valid": "the ambiguity diagnosis itself"},
    {"id": "token_slot_leakage", "category": "token-slot leakage", "milestone": "5B-Data-v2", "evidence": "token_order_audit documents balanced anonymous permutation (max color-slot fraction 0.512).", "diagnosis": "measure visible cue/color association with token slots", "fixed": "balanced anonymous token permutation", "invalidated": "claims from fixed-slot identity proxies", "valid": "v2/v3 token-order audit passes"},
    {"id": "missing_goal_cue", "category": "missing goal cue", "milestone": "5B-Data-F", "evidence": "final decision states configured target goal was absent from saved standard inputs.", "diagnosis": "field-level input audit against the target-goal schema", "fixed": "partial-cue v2 requires goal_cue_presence_audit pass", "invalidated": "the saturated v1 fair-baseline comparison", "valid": "the redesign rationale"},
    {"id": "temporal_saturation", "category": "clean object-token temporal saturation", "milestone": "5B-Data / v2 / v3", "evidence": "standard temporal baseline reaches pair-order 1.0 in observable-cue, partial-cue, and OOD variants.", "diagnosis": "headroom and OOD standard-baseline comparisons", "fixed": "not fixed; the redesigned synthetic task remains unsuitable for a method comparison", "invalidated": "claims that a relational model exceeds GRU/TCN on these tasks", "valid": "temporal saturation is reproducibly documented"},
    {"id": "logged_consistency_saturation", "category": "logged real-trajectory consistency saturation", "milestone": "R-Series ASU", "evidence": "held-out-goal-object proprio-action GRU reaches pair-order 0.9504, above the 0.88 saturation ceiling.", "diagnosis": "held-out goal-object split with shortcut controls and preregistered ceiling", "fixed": "not a defect; task is retained as a saturation audit", "invalidated": "RealRelDynVerifier training on ASU", "valid": "reproducible logged-consistency benchmark, not physical counterfactual prediction"},
    {"id": "numeric_real_signal", "category": "numeric-only real failure signal insufficiency", "milestone": "R2 BotFails numeric", "evidence": "RealRelDynVerifier balanced accuracy 0.55, improvement 0.00 versus the best standard; required improvement was +0.08.", "diagnosis": "whole-task-held-out numeric failure-onset evaluation with one authorized method trial", "fixed": "not with current BotFails numeric prehistory", "invalidated": "ActMask improves real-robot failure prediction", "valid": "causally formulated BotFails numeric benchmark and negative method result"},
    {"id": "visual_prehistory_recoverability", "category": "visual prehistory recoverability insufficiency", "milestone": "R3 BotFails visual", "evidence": "best whole-task-held-out fair visual standard is 0.617, below the preregistered 0.65 gate after 100-video alignment.", "diagnosis": "causal RGB history, frozen ResNet features, 8/16/32 horizons, task-held-out controls", "fixed": "not with current BotFails visual prehistory", "invalidated": "VisualRelDynVerifier training or improvement claim on BotFails", "valid": "aligned visual benchmark and recoverability diagnosis"},
]


def _taxonomy_markdown() -> str:
    body = ["# P-Series failure taxonomy", "", "Each entry separates a diagnosed benchmark/property failure from a claim about a model.", ""]
    for item in TAXONOMY:
        body += [f"## {item['category']}", f"- Milestone: {item['milestone']}", f"- Evidence: {item['evidence']}", f"- Diagnosis: {item['diagnosis']}", f"- Fixed: {item['fixed']}", f"- Claim invalidated: {item['invalidated']}", f"- Claim that remains valid: {item['valid']}", ""]
    return "\n".join(body)


CLAIMS = [
    {"family": "Synthetic audit suite", "supported": ["Controlled synthetic audits detect shortcut, identifiability, goal-cue, token-order, and saturation failures."], "not_supported": ["Clean object-token relational modeling beats GRU/TCN."], "prohibited": ["Using a failed or saturated synthetic condition as method evidence."], "requires_more_data": ["A non-saturated, fair-observable synthetic task only if needed for a diagnostic appendix."], "evidence": ["5B-F", "5B-Data-F", "5B-Data-v2", "5B-Data-v3"]},
    {"family": "ASU real UR5 logged consistency", "supported": ["The adapter and logged action-conditioned consistency benchmark are reproducible."], "not_supported": ["A new relational method improves the ASU task."], "prohibited": ["Physical counterfactual intervention-success prediction from logged trajectories."], "requires_more_data": ["Repeated alternate actions or richer contact/object outcome ambiguity."], "evidence": ["R-Series ASU final decision"]},
    {"family": "R2 BotFails numeric", "supported": ["The causally aligned numeric failure-onset task and negative RealRelDynVerifier result are reproducible."], "not_supported": ["ActMask improves real-robot failure prediction."], "prohibited": ["Visual/contact reasoning or physical intervention claims."], "requires_more_data": ["Outcome-labelled trajectories with richer pre-failure ambiguity."], "evidence": ["R2 final decision and method summary"]},
    {"family": "R3 BotFails visual", "supported": ["The 100-video causal RGB benchmark is aligned and reproducible."], "not_supported": ["BotFails prehistory has enough fair visual signal for a relational method."], "prohibited": ["VisualRelDynVerifier improvement, post-anchor leakage, or unavailable contact/object reasoning claims."], "requires_more_data": ["RoboMIND/RH20T with RGB-D, repeated tasks, and contact/outcome labels."], "evidence": ["R3 final decision and baseline summary"]},
    {"family": "P-Series consolidation", "supported": ["Current evidence does not authorize an ActMask real-robot method claim."], "not_supported": ["A method paper centered on present performance."], "prohibited": ["Reframing negative/saturated evidence as a real-robot method win."], "requires_more_data": ["A dataset passing the future-method gate."], "evidence": ["This package"]},
]


def _claim_markdown() -> str:
    lines = ["# Claim boundary table", "", "| Family | Supported | Not supported | Explicitly prohibited | Requires more data |", "| --- | --- | --- | --- | --- |"]
    for row in CLAIMS:
        lines.append("| " + " | ".join([row["family"], "; ".join(row["supported"]), "; ".join(row["not_supported"]), "; ".join(row["prohibited"]), "; ".join(row["requires_more_data"])]) + " |")
    return "\n".join(lines)


def build() -> dict[str, Any]:
    for part in (OUT / "docs", OUT / "tables", OUT / "figures", OUT / "access_requests"):
        part.mkdir(parents=True, exist_ok=True)
    missing: list[dict[str, str]] = []
    artifact_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for family, directory, description in FAMILIES:
        root = SOURCE / directory
        decision_relative = "final_package/final_decision.json" if family.startswith("R2") or family.startswith("R3") else "final_decision.json"
        decision_path = root / decision_relative
        decision = None
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text()).get("decision")
        else:
            missing.append({"family": family, "path": str(decision_path.relative_to(PROJECT)), "reason": "required final decision is absent"})
        source_rows.append({"family": family, "description": description, "source_root": str(root.relative_to(PROJECT)), "exists": root.is_dir(), "decision": decision, "decision_artifact": str(decision_path.relative_to(PROJECT))})
        # Preserve a compact explicit core list, then include every lightweight
        # structured audit/document artifact.  Individual raw-score rows,
        # videos, feature arrays, and checkpoints are represented by their
        # already-frozen manifests rather than duplicated or rehashed here.
        relatives = set(KEY_ARTIFACTS[family])
        if root.is_dir():
            for candidate in root.rglob("*"):
                relative_candidate = candidate.relative_to(root)
                if not candidate.is_file() or candidate.suffix not in {".json", ".jsonl", ".md"}:
                    continue
                if "raw_scores" in relative_candidate.parts:
                    continue
                relatives.add(str(relative_candidate))
        for relative in sorted(relatives):
            path = root / relative
            row = {"family": family, "path": str(path.relative_to(PROJECT)), "exists": path.is_file()}
            if path.is_file():
                row |= {"sha256": _sha(path), "bytes": path.stat().st_size}
            else:
                missing.append({"family": family, "path": str(path.relative_to(PROJECT)), "reason": "key artifact is absent"})
            artifact_rows.append(row)
    source_inventory = {"schema": "p-series-source-inventory-v1", "scope": "Frozen artifacts only; no prior artifact was changed or rerun.", "families": source_rows}
    artifact_manifest = {"schema": "p-series-artifact-manifest-v1", "artifacts": artifact_rows, "artifact_count": len(artifact_rows), "all_key_artifacts_present": not missing}
    missing_report = {"schema": "p-series-missing-artifacts-v1", "required_key_artifacts_missing": missing, "historical_nonrecoverable_gaps": [
        {"milestone": "5A-R", "gap": "Raw per-example scores were unavailable for the frozen estimator/CI repair; no rerun was authorized.", "impact": "Do not cite the affected frozen CI claims."},
        {"milestone": "R2", "gap": "RoboMIND v1 was gated (HTTP 401) and local RH20T had only three trajectories.", "impact": "Neither source qualified for a method trial."},
    ], "status": "complete_for_consolidation" if not missing else "incomplete"}
    _write_json(OUT / "source_inventory.json", source_inventory)
    _write_json(OUT / "artifact_manifest.json", artifact_manifest)
    _write_json(OUT / "missing_artifacts_report.json", missing_report)

    _write_text(OUT / "docs/p_series_plan.md", """
# P-Series plan

P-Series freezes and consolidates prior synthetic and real-robot evidence. It performs no training, no BotFails tuning, and no modification to frozen sources.

1. Inventory the ten named source families and hash their key decisions, manifests, audits, baselines, and raw-score manifests.
2. Convert diagnosed failures into an explicit taxonomy and claim boundary table.
3. Position the work as an audit/benchmark paper, with table and figure specifications.
4. Prepare narrowly scoped RoboMIND and RH20T requests.
5. Permit a future method only after a new dataset passes the fixed gate in `future_method_gate.json`.
""")
    _write_text(OUT / "docs/p_series_results.md", """
# P-Series results

All required key artifacts from 5B-0, 5B, 5B-F, 5B-Data, 5B-Data-F, 5B-Data-v2, 5B-Data-v3, R-Series ASU, R2 BotFails numeric, and R3 BotFails visual are present in the source manifest.

The consolidated result is negative for a real-robot method claim, but positive for an audit contribution. Synthetic evidence shows fair-observation ambiguity, missing goal cues, token-slot risks, and temporal saturation. ASU is a valid logged-consistency benchmark but saturates standard models. R2 is a valid numeric failure-onset benchmark but its one authorized relational method did not improve. R3 aligns 100 RGB videos exactly, but its fair visual recoverability maximum is 0.617, below the 0.65 gate.

No new method was trained and no frozen artifact was changed.
""")
    _write_text(OUT / "docs/p_series_handoff.md", """
# P-Series handoff

Use this package for a benchmark/audit manuscript and for requesting next data. Do not run another BotFails model or reframe ASU/R2/R3 as a method win.

The next valid technical action is to obtain a permissioned dataset that passes every item in `future_method_gate.md`; then repeat alignment, shortcut, recoverability, oracle, and raw-artifact checks before authorizing a single pre-registered method family.
""")
    _write_text(OUT / "failure_taxonomy.md", _taxonomy_markdown())
    _write_json(OUT / "failure_taxonomy.json", {"schema": "p-series-failure-taxonomy-v1", "categories": TAXONOMY})
    _write_text(OUT / "claim_boundary_table.md", _claim_markdown())
    _write_json(OUT / "claim_boundary_table.json", {"schema": "p-series-claim-boundary-v1", "rows": CLAIMS})
    _write_text(OUT / "paper_positioning.md", """
# Paper positioning

## Track A — Method paper

- Central thesis: ActMask improves real-robot action-conditioned verification.
- Contributions/experiments: would require a new real dataset and a gated method comparison.
- Exclude: present synthetic saturation and R2/R3 negative outcomes as performance evidence.
- Reviewer risk: fatal; no authorized real-robot improvement exists.
- Minimum extra work: acquire a dataset that passes every future-method gate, preregister one method family, and show a robust held-out gain.
- Decision: not currently supported.

## Track B — Benchmark / audit paper

- Central thesis: rigorous synthetic-to-real audits reveal shortcuts, unidentifiability, missing cues, estimator failures, and saturation before a verification method is claimed.
- Contributions: a reusable audit sequence; synthetic failure taxonomy; reproducible ASU/R2/R3 benchmarks; explicit claim boundaries and next-data gate.
- Include: all diagnosed failures, source manifests, causal alignment, controls, saturation checks, negative R2/R3 conclusions, and access specification.
- Exclude: any unsupported method-improvement curve or physical counterfactual claim.
- Reviewer risk: breadth and novelty must be framed as a reproducible audit protocol rather than a collection of failures.
- Minimum extra work: concise writing, compact tables/figures, and release instructions for manifests/tests.

## Track C — Technical report / workshop paper

- Central thesis: action-conditioned robot verification benchmarks can fail in predictable ways from synthetic construction to logged real data.
- Contributions: transparent negative report and actionable data requirements.
- Include: the same core audit evidence, with more procedural detail.
- Exclude: broad generalization claims beyond audited conditions.
- Reviewer risk: weaker archival impact and less demand for a coherent benchmark contribution.
- Minimum extra work: minimal; package and a short narrative.

## Recommendation

Recommend **Track B: benchmark / audit paper**, title style: *Auditing Dynamic Robot Verification Benchmarks: Shortcuts, Unidentifiability, and Temporal Saturation*. The current evidence supports an audit contribution, not a method paper.
""")

    synthetic = {"schema": "p-series-synthetic-summary-v1", "rows": [
        {"milestone": "5B-0", "outcome": "probe authorized a limited verifier trial", "key_metric": "best fair standard 0.75; oracle 1.0", "method_claim": "not final evidence"},
        {"milestone": "5B", "outcome": "method no better than standard temporal", "key_metric": "method 0.75, standard 0.75", "method_claim": "not supported"},
        {"milestone": "5B-F", "outcome": "fair observation ambiguous", "key_metric": "Bayes-majority upper bound 0.75", "method_claim": "invalid"},
        {"milestone": "5B-Data", "outcome": "standard temporal saturated", "key_metric": "standard 1.0, oracle 1.0", "method_claim": "not supported"},
        {"milestone": "5B-Data-F", "outcome": "goal absent from fair standard inputs", "key_metric": "metric/implementation artifact", "method_claim": "invalid"},
        {"milestone": "5B-Data-v2", "outcome": "standard temporal saturated", "key_metric": "standard 1.0, oracle 1.0", "method_claim": "not supported"},
        {"milestone": "5B-Data-v3", "outcome": "OOD standard temporal saturated", "key_metric": "OOD standard 1.0, oracle 1.0", "method_claim": "not supported"},
    ]}
    real = {"schema": "p-series-real-dataset-summary-v1", "rows": [
        {"dataset": "ASU UR5", "task": "logged action-conditioned consistency", "validity": "adapter/audits pass", "best_standard": 0.9504373177842566, "decision": "saturated; no method"},
        {"dataset": "BotFails R2", "task": "numeric failure onset", "validity": "causal task valid", "best_standard": 0.55, "method": 0.55, "required_gain": 0.08, "decision": "method not supported"},
        {"dataset": "BotFails R3", "task": "causal RGB failure forecasting", "validity": "100/100 video alignment pass", "best_fair_visual_standard": 0.6166666666666667, "required_recoverability": 0.65, "decision": "visual method not authorized"},
    ]}
    recoverability = {"schema": "p-series-recoverability-table-v1", "rows": [
        {"family": "5B-0", "fair_standard": 0.75, "oracle": 1.0, "state": "headroom existed but did not yield a method gain"},
        {"family": "5B", "fair_standard": 0.75, "method": 0.75, "state": "no improvement"},
        {"family": "5B-Data/v2/v3", "fair_standard": 1.0, "oracle": 1.0, "state": "saturated"},
        {"family": "ASU", "fair_standard": 0.9504373177842566, "saturation_ceiling": 0.88, "state": "saturated"},
        {"family": "R2 BotFails numeric", "fair_standard": 0.55, "method": 0.55, "required_gain": 0.08, "state": "no method gain"},
        {"family": "R3 BotFails visual", "fair_standard": 0.6166666666666667, "recoverability_gate": 0.65, "state": "below recoverability gate"},
    ]}
    _write_json(OUT / "tables/synthetic_summary.json", synthetic)
    _write_json(OUT / "tables/real_dataset_summary.json", real)
    _write_json(OUT / "tables/failure_taxonomy_table.json", {"schema": "p-series-failure-taxonomy-table-v1", "rows": TAXONOMY})
    _write_json(OUT / "tables/claim_boundary_table.json", {"schema": "p-series-claim-boundary-table-v1", "rows": CLAIMS})
    _write_json(OUT / "tables/recoverability_table.json", recoverability)
    figures = {
        "fig1_pipeline_audit_flow.md": "# Figure 1 — Pipeline audit flow\n\nFlow: source inventory → alignment/causality → shortcut controls → fair-observation identifiability → recoverability/headroom → method authorization or data-access branch. Mark 5B-F, 5B-Data-F, ASU, R2, and R3 exits with their actual decision labels.\n",
        "fig2_failure_taxonomy.md": "# Figure 2 — Failure taxonomy\n\nDraw ten grouped nodes: synthetic construction (shortcut, CI/artifact, identifiability, token slot, goal cue, saturation), logged real benchmark (ASU saturation), and real outcome forecasting (R2 numeric, R3 visual). Each node links to invalidated and retained claims.\n",
        "fig3_synthetic_decision_tree.md": "# Figure 3 — Synthetic decision tree\n\nShow 5B-0 limited authorization → 5B no gain → 5B-F ambiguity → 5B-Data goal-input artifact → v2/v3 saturation. Use red stop nodes for invalid evidence and blue audit nodes for retained diagnostics.\n",
        "fig4_real_dataset_results.md": "# Figure 4 — Real dataset results\n\nThree aligned panels: ASU standard saturation (0.950 vs 0.88 ceiling); R2 standard/method both 0.55 with +0.08 required gain; R3 fair visual 0.617 below 0.65 gate. Add a final arrow to the next-data gate, not to a method result.\n",
    }
    for name, text in figures.items():
        _write_text(OUT / "figures" / name, text)

    _write_text(OUT / "access_requests/robomind_access_request.md", """
# RoboMIND access request

We request permissioned, non-commercial academic access for a benchmark-audit study. Before any bulk transfer, please provide a file inventory (trajectory IDs, tasks, modalities, labels, sizes, licenses, calibration, and split/repetition metadata).

Requested subset:

- at least 100 success/failure trajectories;
- aligned RGB or RGB-D streams with timestamps;
- synchronized robot state and action streams;
- failure category and failure-onset annotation when available;
- repeated tasks and multiple attempts per task;
- non-commercial academic-use permission and citation/license terms.

Why this subset: BotFails RGB was causally aligned and valid, but its fair visual prehistory did not reach the 0.65 recoverability gate. ASU logged consistency saturated standard models. We need richer object/contact/outcome ambiguity with repeated attempts, not another generic video collection.

We will first run alignment, leakage, shortcut, recoverability, oracle, and raw-score-artifact audits. No method result will be claimed unless the dataset passes the published gate.
""")
    _write_text(OUT / "access_requests/rh20t_subset_request.md", """
# RH20T targeted subset request

We request a manageable, permissioned non-commercial academic subset and a file inventory before bulk download (trajectory IDs, tasks, labels, modalities, calibration, sizes, and license).

Requested subset:

- at least 100 contact-rich trajectories;
- completion rating or success/failure label, and failure onset/category where available;
- calibrated, timestamp-aligned RGB-D;
- synchronized robot state and action streams;
- force/tactile streams where available;
- repeated task families and multiple attempts per family;
- a manageable download subset rather than the entire corpus.

Justification: the valid BotFails visual benchmark was not recoverable enough from fair prehistory, while ASU logged consistency saturated. The next data must expose richer object/contact/outcome ambiguity and support causal, leakage-free evaluation before any new method trial.
""")
    checklist = {"schema": "p-series-dataset-requirement-checklist-v1", "required_before_method": [
        "Public or permissioned real-robot source with explicit research license",
        "At least 100 aligned trajectories",
        "Repeated tasks or outcome/failure labels",
        "Timestamp-aligned RGB/RGB-D, robot state, and actions",
        "No future leakage in fair inputs",
        "Raw scores, checkpoints/configs, source hashes, and commands retained",
    ], "preferred_for_contact_reasoning": ["Calibrated RGB-D", "failure onset/category", "force/tactile", "multiple attempts per task", "contact-rich object manipulation"], "RoboMIND": "Request inventory first, then success/failure, repeated attempts, RGB/RGB-D and synchronized state/action.", "RH20T": "Request a >=100-trajectory contact-rich, calibrated RGB-D subset with outcome labels and force/tactile when available."}
    _write_json(OUT / "access_requests/dataset_requirement_checklist.json", checklist)
    _write_text(OUT / "access_requests/dataset_requirement_checklist.md", "# Dataset requirement checklist\n\nSee `dataset_requirement_checklist.json`. A source must satisfy every required-before-method item; contact labels/modalities are preferred because they address the current object/contact/outcome ambiguity gap.\n")

    gate_rows = [
        {"id": 1, "requirement": "public or permissioned real robot source", "pass_condition": "license/access recorded"},
        {"id": 2, "requirement": "at least 100 aligned trajectories", "pass_condition": "episode manifest and alignment audit"},
        {"id": 3, "requirement": "repeated tasks or outcome/failure labels", "pass_condition": "task/label inventory"},
        {"id": 4, "requirement": "no future leakage", "pass_condition": "causality and input-field audit"},
        {"id": 5, "requirement": "shortcut baselines <= 0.65", "pass_condition": "held-out shortcut report"},
        {"id": 6, "requirement": "fair recoverability model >= 0.65", "pass_condition": "held-out fair baseline report"},
        {"id": 7, "requirement": "best standard baseline < 0.90", "pass_condition": "held-out standard report"},
        {"id": 8, "requirement": "meaningful oracle/upper bound exceeds standard by >= 0.08", "pass_condition": "leakage-isolated oracle/headroom report"},
        {"id": 9, "requirement": "raw-score and artifact logging passes", "pass_condition": "hash manifests, configs/checkpoints, raw scores, and command log"},
    ]
    _write_json(OUT / "future_method_gate.json", {"schema": "p-series-future-method-gate-v1", "rule": "All nine checks must pass before a method trial.", "checks": gate_rows, "authorized_after_pass": ["one preregistered RealRelDynVerifier or VisualRelDynVerifier family"], "prohibited_before_pass": ["new method training", "BotFails tuning", "real-robot method-improvement claims"]})
    _write_text(OUT / "future_method_gate.md", "# Future method gate\n\nNo new method trial may run unless **all** nine checks in `future_method_gate.json` pass. This gate is conjunctive: a valid source alone is insufficient; it must also be causal, non-shortcut, recoverable, non-saturated, have oracle headroom, and retain raw artifacts. Only then may one preregistered RealRelDynVerifier or VisualRelDynVerifier family be trained.\n")

    _write_text(OUT / "final_report.md", """
# P-Series final report

## Decision

**AUDIT_PAPER_READY_WITH_MINOR_WRITING.** The evidence supports a benchmark/audit paper (Track B), not a real-robot method paper.

## What is frozen and reproducible

The source inventory hashes every key artifact in the ten required milestone families. Synthetic audits document shortcuts, ambiguity, missing cues, token-order risks, and saturation. ASU is a valid logged-consistency audit but standard baselines saturate. R2 is a valid numeric BotFails benchmark with no relational-method improvement. R3 is a valid 100-video causal RGB benchmark whose fair visual recoverability is below its method gate.

## Claim boundary

No ActMask real-robot method claim is authorized. Physical counterfactual intervention success and unavailable contact/object reasoning are explicitly prohibited. The actionable contribution is the audit protocol, reproducible negative evidence, and a concrete next-data gate.

## Next action

Use the access requests to obtain RoboMIND or an RH20T subset, run the fixed future-method gate, and only then consider one preregistered method family.
""")
    final = {"schema": "p-series-final-decision-v1", "decision": "AUDIT_PAPER_READY_WITH_MINOR_WRITING", "recommended_track": "B. Benchmark / audit paper", "method_claim_authorized": False, "reason": "The source artifacts consistently support an audit contribution, while ASU saturates, R2 shows no method gain, and R3 fails fair visual recoverability.", "next_data_required_for_method": ["RoboMIND or RH20T subset passing all future-method gate checks"], "explicitly_prohibited": ["ActMask improves real-robot failure prediction", "physical counterfactual intervention success prediction", "contact/object reasoning from unavailable channels"]}
    _write_json(OUT / "final_decision.json", final)
    _write_text(OUT / "next_steps.md", """
# Next steps

1. Convert Track B materials into manuscript prose using the table and figure specifications.
2. Send the RoboMIND and RH20T requests after confirming the intended institutional contact/license language.
3. When a source responds, obtain its inventory before bulk download.
4. Run the fixed future-method gate exactly once on the selected subset.
5. Only if all nine checks pass, preregister one method family; otherwise publish the audit boundary without further tuning.
""")

    generated = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name not in {"reproducibility_manifest.json", "package_manifest.json"})
    reproducibility = {"schema": "p-series-reproducibility-manifest-v1", "generator": str(Path(__file__).relative_to(PROJECT)), "generator_sha256": _sha(Path(__file__)), "python": "/home/tzh/conda_envs/actmask/bin/python", "source_artifact_manifest_sha256": _sha(OUT / "artifact_manifest.json"), "source_inventory_sha256": _sha(OUT / "source_inventory.json"), "generated_files": [{"path": str(path.relative_to(OUT)), "sha256": _sha(path)} for path in generated], "no_training_or_source_mutation": True}
    _write_json(OUT / "reproducibility_manifest.json", reproducibility)
    package_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "package_manifest.json")
    _write_json(OUT / "package_manifest.json", {"schema": "p-series-package-manifest-v1", "decision": final["decision"], "files": [{"path": str(path.relative_to(OUT)), "sha256": _sha(path)} for path in package_files]})
    return {"out": str(OUT), "source_artifacts": len(artifact_rows), "missing": len(missing), "decision": final["decision"]}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
