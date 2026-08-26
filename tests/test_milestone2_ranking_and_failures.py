"""Focused exact tests for M2 candidate ranking and failure evidence."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from actmask.eval.prediction import PredictionRecord
from actmask.eval.ranking import action_ranking_records, build_candidate_sets
from actmask.visualization.milestone2_failures import (
    FAILURE_CATEGORIES,
    generate_failure_visualizations,
    select_failure_cases,
    write_selection_manifest,
)


def _record(
    *,
    sample_id: int,
    probabilities: list[float],
    targets: list[int],
    success: float,
    pair_id: int | None = None,
    variant_id: int | None = None,
    counterfactual_type: str = "none",
    scenario: str = "default",
    distribution: str = "ID",
    ood_axis: str = "none",
    metadata: dict[str, object] | None = None,
) -> PredictionRecord:
    count = len(probabilities)
    points = torch.stack(
        (
            torch.linspace(0.0, 0.3, count),
            torch.linspace(0.1, 0.4, count),
            torch.linspace(-0.1, 0.2, count),
        ),
        dim=1,
    )
    return PredictionRecord(
        probabilities=torch.tensor(probabilities, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        sample_id=sample_id,
        pair_id=pair_id,
        variant_id=variant_id,
        scenario=scenario,
        success=success,
        counterfactual_type=counterfactual_type,
        distribution=distribution,
        ood_axis=ood_axis,
        metadata=dict(metadata or {}),
        points=points,
        velocities=torch.full((count, 3), 0.06, dtype=torch.float32),
        action=torch.tensor([0.0, 0.0, 0.0, 0.4, 0.1, 0.2, 1.0, 0.08]),
    )


def _ranking_records() -> list[PredictionRecord]:
    # The failed action has the highest model score.  Of two successful actions,
    # only one is retrieved in top 3.  All values are exact under the tie-aware
    # pairwise and AUC definitions.
    rows = [
        ("a_failed", 0.90, 0.0),
        ("b_success", 0.80, 1.0),
        ("c_failed", 0.30, 0.0),
        ("d_success", 0.20, 1.0),
    ]
    return [
        _record(
            sample_id=index,
            probabilities=[score] * 4,
            targets=[1, 0, 0, 0],
            success=success,
            metadata={"candidate_set_id": "scene-A", "candidate_id": candidate_id},
        )
        for index, (candidate_id, score, success) in enumerate(rows, start=1)
    ]


def test_candidate_action_ranking_metrics_are_exact_and_order_independent() -> None:
    records = _ranking_records()
    rows = action_ranking_records(
        records,
        method_id="Full",
        seed=1301,
        split="id_test",
    )
    aggregate, scene = rows

    assert aggregate["row_type"] == "action_ranking_aggregate"
    assert aggregate["ranking_scene_count"] == 1
    assert aggregate["candidate_count"] == 4
    assert aggregate["top1_successful_action_accuracy"] == pytest.approx(0.0)
    assert aggregate["top3_success_recall"] == pytest.approx(0.5)
    assert aggregate["top3_success_hit"] == pytest.approx(1.0)
    assert aggregate["pairwise_ranking_accuracy"] == pytest.approx(0.25)
    assert aggregate["pairwise_ranking_comparisons"] == 4
    assert aggregate["binary_ranking_auc"] == pytest.approx(0.25)
    assert aggregate["global_binary_ranking_auc"] == pytest.approx(0.25)
    assert aggregate["oracle_regret"] == pytest.approx(1.0)
    assert scene["top1_candidate_id"] == "a_failed"
    assert scene["oracle_candidate_id"] == "b_success"
    assert scene["score_reduction"] == "mean_mask_probability"

    # Input order does not change rankings or report records.
    reordered = action_ranking_records(list(reversed(records)), method_id="Full", seed=1301)
    assert rows == reordered


def test_candidate_utility_is_not_shadowed_by_scene_oracle_utility() -> None:
    """Regret/pairwise metrics must use the selected candidate's utility.

    The ranking dataset records ``oracle_utility`` as the scene-wide maximum
    on every candidate, while ``candidate_utility`` is the per-action target
    contact fraction.  Treating the former as each candidate's utility would
    silently force every pairwise comparison and regret to zero.
    """

    records = [
        _record(
            sample_id=101,
            probabilities=[0.95, 0.95],
            targets=[0, 0],
            success=0.0,
            metadata={
                "candidate_set_id": "scene-utility",
                "candidate_id": "failed",
                "candidate_utility": 0.0,
                "oracle_utility": 1.0,
                "oracle_candidate_id": "successful",
            },
        ),
        _record(
            sample_id=102,
            probabilities=[0.10, 0.10],
            targets=[1, 0],
            success=1.0,
            metadata={
                "candidate_set_id": "scene-utility",
                "candidate_id": "successful",
                "candidate_utility": 1.0,
                "oracle_utility": 1.0,
                "oracle_candidate_id": "successful",
                "is_oracle_candidate": True,
            },
        ),
    ]

    aggregate, scene = action_ranking_records(records, method_id="Full", seed=1301)
    assert scene["oracle_candidate_id"] == "successful"
    assert scene["top1_candidate_id"] == "failed"
    assert scene["pairwise_ranking_comparisons"] == 1
    assert scene["pairwise_ranking_accuracy"] == pytest.approx(0.0)
    assert scene["oracle_regret"] == pytest.approx(1.0)
    assert aggregate["oracle_regret"] == pytest.approx(1.0)


def test_score_ties_are_tie_aware_not_candidate_id_shortcuts() -> None:
    """An action-independent scorer cannot win by the successful ID being first.

    The tiny score offsets emulate float32 reduction-order variation across
    batches.  They lie inside the documented 1e-6 tie band and must produce
    the same expected metrics regardless of record/input order.
    """

    records = [
        _record(
            sample_id=index,
            probabilities=[0.5 + offset] * 4,
            targets=[1 if success else 0, 0, 0, 0],
            success=success,
            metadata={
                "candidate_set_id": "all-tied",
                "candidate_id": candidate_id,
                "candidate_utility": float(success),
                "oracle_utility": 1.0,
                "oracle_candidate_id": "a_success",
                "is_oracle_candidate": bool(success),
            },
        )
        for index, (candidate_id, success, offset) in enumerate(
            (
                ("a_success", 1.0, 0.0),
                ("b_failed", 0.0, 5.0e-8),
                ("c_failed", 0.0, -4.0e-8),
                ("d_failed", 0.0, 3.0e-8),
            ),
            start=1,
        )
    ]

    rows = action_ranking_records(records, method_id="NoAction", seed=1301)
    aggregate, scene = rows
    # The display representative is the raw highest-scoring record, but it is
    # explicitly not used for a reported action-selection metric because all
    # four values fall inside the score-tie tolerance.
    assert scene["top1_candidate_id"] == "b_failed"
    assert scene["top1_display_success"] == pytest.approx(0.0)
    assert scene["top1_tie_count"] == 4
    assert scene["tie_policy"] == "uniform_expected_within_score_tolerance"
    assert aggregate["top1_successful_action_accuracy"] == pytest.approx(0.25)
    assert aggregate["top3_success_recall"] == pytest.approx(0.75)
    assert aggregate["top3_success_hit"] == pytest.approx(0.75)
    assert aggregate["pairwise_ranking_accuracy"] == pytest.approx(0.5)
    assert aggregate["ranking_auc"] == pytest.approx(0.5)
    assert aggregate["oracle_regret"] == pytest.approx(0.75)
    assert rows == action_ranking_records(
        list(reversed(records)), method_id="NoAction", seed=1301
    )

    # Renaming the candidates changes only provenance/display identifiers;
    # a successful candidate cannot be promoted by a lexicographically early
    # ID inside a score-tolerance tie.
    renamed = deepcopy(records)
    renamed_ids = {
        "a_success": "z_success",
        "b_failed": "a_failed",
        "c_failed": "m_failed",
        "d_failed": "q_failed",
    }
    for record in renamed:
        original_id = str(record.metadata["candidate_id"])
        record.metadata["candidate_id"] = renamed_ids[original_id]
        record.metadata["oracle_candidate_id"] = "z_success"
    renamed_aggregate, renamed_scene = action_ranking_records(
        renamed, method_id="NoAction", seed=1301
    )
    metric_fields = (
        "top1_successful_action_accuracy",
        "top3_success_recall",
        "top3_success_hit",
        "pairwise_ranking_accuracy",
        "ranking_auc",
        "oracle_regret",
    )
    for field in metric_fields:
        assert renamed_aggregate[field] == pytest.approx(aggregate[field])
        assert renamed_scene[field] == pytest.approx(scene[field])

    # With literally equal scores the lexical ID changes the display panel
    # representative from success to failure.  The reported tie-aware action
    # metrics must nevertheless remain identical.
    exact_scores = deepcopy(records)
    for record in exact_scores:
        record.probabilities.fill_(0.5)
    exact_aggregate, exact_scene = action_ranking_records(
        exact_scores, method_id="NoAction", seed=1301
    )
    exact_renamed = deepcopy(exact_scores)
    for record in exact_renamed:
        original_id = str(record.metadata["candidate_id"])
        record.metadata["candidate_id"] = renamed_ids[original_id]
        record.metadata["oracle_candidate_id"] = "z_success"
    exact_renamed_aggregate, exact_renamed_scene = action_ranking_records(
        exact_renamed, method_id="NoAction", seed=1301
    )
    assert exact_scene["top1_display_success"] == pytest.approx(1.0)
    assert exact_renamed_scene["top1_display_success"] == pytest.approx(0.0)
    for field in metric_fields:
        assert exact_renamed_aggregate[field] == pytest.approx(exact_aggregate[field])
        assert exact_renamed_scene[field] == pytest.approx(exact_scene[field])


def test_candidate_sets_reject_duplicate_ids_and_missing_success() -> None:
    duplicate = _ranking_records()
    duplicate[1].metadata["candidate_id"] = "a_failed"
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        build_candidate_sets(duplicate)

    no_success = _ranking_records()
    for record in no_success:
        record.success = 0.0
    with pytest.raises(ValueError, match="no successful action"):
        build_candidate_sets(no_success)

    # Oracle provenance must describe one real candidate rather than letting
    # candidate-ID ordering silently choose between conflicting declarations.
    missing_oracle = _ranking_records()
    for record in missing_oracle:
        record.metadata["oracle_candidate_id"] = "not-in-scene"
    with pytest.raises(ValueError, match="not a candidate"):
        build_candidate_sets(missing_oracle)

    duplicate_markers = _ranking_records()
    for record in duplicate_markers[:2]:
        record.metadata["is_oracle_candidate"] = True
    with pytest.raises(ValueError, match="more than one is_oracle_candidate"):
        build_candidate_sets(duplicate_markers)

    # Counterfactual pair/group IDs alone are not valid ranking metadata: this
    # prevents the existing two-variant diagnostic from being misreported as a
    # multi-candidate action-selection benchmark.
    legacy_pair = _record(
        sample_id=88,
        probabilities=[0.8] * 4,
        targets=[1, 0, 0, 0],
        success=1.0,
        pair_id=9,
        variant_id=0,
    )
    with pytest.raises(ValueError, match="missing scene key"):
        build_candidate_sets([legacy_pair])


def _failure_records() -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    actmask: list[PredictionRecord] = []
    geometric: list[PredictionRecord] = []

    def append_pair(
        *,
        sample_id: int,
        act: list[float],
        geo: list[float],
        target: list[int],
        success: float,
        pair_id: int | None = None,
        variant_id: int | None = None,
        counterfactual_type: str = "none",
        scenario: str = "default",
        distribution: str = "ID",
        ood_axis: str = "none",
        metadata: dict[str, object] | None = None,
    ) -> None:
        actmask.append(
            _record(
                sample_id=sample_id,
                probabilities=act,
                targets=target,
                success=success,
                pair_id=pair_id,
                variant_id=variant_id,
                counterfactual_type=counterfactual_type,
                scenario=scenario,
                distribution=distribution,
                ood_axis=ood_axis,
                metadata=metadata,
            )
        )
        geometric.append(
            _record(
                sample_id=sample_id,
                probabilities=geo,
                targets=target,
                success=success,
                pair_id=pair_id,
                variant_id=variant_id,
                counterfactual_type=counterfactual_type,
                scenario=scenario,
                distribution=distribution,
                ood_axis=ood_axis,
                metadata=metadata,
            )
        )

    # Opposite single-sample wins establish the two model-comparison classes.
    append_pair(
        sample_id=1,
        act=[0.9, 0.1, 0.1, 0.1],
        geo=[0.1, 0.9, 0.1, 0.1],
        target=[1, 0, 0, 0],
        success=1.0,
    )
    append_pair(
        sample_id=2,
        act=[0.1, 0.9, 0.1, 0.1],
        geo=[0.9, 0.1, 0.1, 0.1],
        target=[1, 0, 0, 0],
        success=0.0,
    )
    # Real paired counterfactuals change GT but the ActMask predictions do not.
    for sample_id, pair_id, kind in ((30, 10, "velocity"), (40, 11, "action")):
        append_pair(
            sample_id=sample_id,
            act=[0.1, 0.1, 0.1, 0.1],
            geo=[0.9, 0.1, 0.1, 0.1],
            target=[1, 0, 0, 0],
            success=1.0,
            pair_id=pair_id,
            variant_id=0,
            counterfactual_type=kind,
            scenario=f"{kind}_counterfactual",
        )
        append_pair(
            sample_id=sample_id + 1,
            act=[0.1, 0.1, 0.1, 0.1],
            geo=[0.1, 0.9, 0.1, 0.1],
            target=[0, 1, 0, 0],
            success=0.0,
            pair_id=pair_id,
            variant_id=1,
            counterfactual_type=kind,
            scenario=f"{kind}_counterfactual",
        )
    append_pair(
        sample_id=50,
        act=[0.1, 0.9, 0.1, 0.1],
        geo=[0.9, 0.1, 0.1, 0.1],
        target=[1, 0, 0, 0],
        success=0.0,
        distribution="OOD",
        ood_axis="unseen_velocity_magnitude",
        metadata={"domain": "ood"},
    )
    # A real candidate set whose top-one action is failed, though a success is
    # available, supplies the ranking failure class.
    append_pair(
        sample_id=60,
        act=[0.95, 0.95, 0.95, 0.95],
        geo=[0.1, 0.1, 0.1, 0.1],
        target=[0, 0, 0, 0],
        success=0.0,
        metadata={"candidate_set_id": "rank-scene", "candidate_id": "failed"},
    )
    append_pair(
        sample_id=61,
        act=[0.05, 0.05, 0.05, 0.05],
        geo=[0.9, 0.1, 0.1, 0.1],
        target=[1, 0, 0, 0],
        success=1.0,
        metadata={"candidate_set_id": "rank-scene", "candidate_id": "success"},
    )
    return actmask, geometric


def test_failure_selection_and_rendering_cover_six_real_categories(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    actmask, geometric = _failure_records()
    selections = select_failure_cases(actmask, geometric)
    assert [selection.category for selection in selections] == list(FAILURE_CATEGORIES)
    assert all(selection.qualifying for selection in selections)

    # Selection is independent of input ordering and carries only IDs/metrics
    # into its manifest, not fabricated scene data.
    reversed_selections = select_failure_cases(list(reversed(actmask)), list(reversed(geometric)))
    assert [item.manifest_record() for item in selections] == [
        item.manifest_record() for item in reversed_selections
    ]
    manifest = write_selection_manifest(root / "manual_manifest.json", selections)
    first_bytes = manifest.read_bytes()
    write_selection_manifest(manifest, selections)
    assert manifest.read_bytes() == first_bytes

    artifacts = generate_failure_visualizations(
        actmask,
        geometric,
        root / "failures",
        manifest_metadata={"seed": 1301, "method": "Full"},
    )
    assert artifacts.manifest_path.is_file()
    assert set(artifacts.paths) == set(FAILURE_CATEGORIES)
    assert all(path.is_file() and path.stat().st_size > 500 for path in artifacts.paths.values())
    payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert [row["category"] for row in payload["selections"]] == list(FAILURE_CATEGORIES)
    assert all(row["qualifying"] is True for row in payload["selections"])
    assert "Ground truth" not in artifacts.manifest_path.read_text(encoding="utf-8")


def test_failure_selector_marks_missing_categories_nonqualifying() -> None:
    good = _record(
        sample_id=99,
        probabilities=[0.9, 0.1, 0.1, 0.1],
        targets=[1, 0, 0, 0],
        success=1.0,
    )
    selections = {selection.category: selection for selection in select_failure_cases([good], [good])}
    assert selections["velocity_counterfactual_pair"].qualifying is False
    assert selections["action_counterfactual_pair"].qualifying is False
    assert selections["ood_failure"].qualifying is False
    assert selections["action_ranking_failure"].qualifying is False
    assert "no complete real counterfactual" in selections["velocity_counterfactual_pair"].reason
    assert "no usable real multi-candidate" in selections["action_ranking_failure"].reason


def test_counterfactual_failure_panels_exclude_ranking_records_with_source_names() -> None:
    """A ranking candidate must never masquerade as a physical CF variant."""

    actmask, geometric = _failure_records()
    # These deliberately look tempting to the old scenario-name heuristic:
    # source pair IDs/scenarios are preserved but neither record is a source
    # velocity/action counterfactual or the variant-0/1 binary pair.
    for offset, scenario in enumerate(("velocity_counterfactual", "action_counterfactual")):
        kwargs = dict(
            sample_id=700 + offset,
            probabilities=[0.1, 0.1, 0.1, 0.1],
            targets=[1, 0, 0, 0],
            success=0.0,
            pair_id=800 + offset,
            variant_id=0,
            counterfactual_type="candidate_action_ranking",
            scenario=scenario,
            metadata={
                "candidate_set_id": f"trap-{offset}",
                "candidate_id": "failed",
                "candidate_utility": 0.0,
                "oracle_utility": 1.0,
            },
        )
        actmask.append(_record(**kwargs))
        geometric.append(_record(**kwargs))

    selections = {item.category: item for item in select_failure_cases(actmask, geometric)}
    for category, allowed in (
        ("velocity_counterfactual_pair", {"velocity", "velocity_and_future_dynamics"}),
        ("action_counterfactual_pair", {"action", "action_timing", "action_radius"}),
    ):
        selection = selections[category]
        assert selection.qualifying is True
        assert len(selection.examples) == 2
        assert {
            str(example.actmask.counterfactual_type) for example in selection.examples
        }.issubset(allowed)
        assert {int(example.actmask.variant_id) for example in selection.examples} == {0, 1}

    # A tied successful/failed ranking set has no strict top-score failure;
    # candidate-ID display order must not manufacture a ranking panel.
    tied = [
        _record(
            sample_id=900 + index,
            probabilities=[0.5, 0.5, 0.5, 0.5],
            targets=[1 if success else 0, 0, 0, 0],
            success=success,
            counterfactual_type="candidate_action_ranking",
            metadata={
                "candidate_set_id": "tied-ranking",
                "candidate_id": candidate_id,
                "candidate_utility": float(success),
                "oracle_utility": 1.0,
            },
        )
        for index, (candidate_id, success) in enumerate(
            (("a_success", 1.0), ("b_failed", 0.0)), start=1
        )
    ]
    tied_selection = {
        item.category: item for item in select_failure_cases(tied, tied)
    }["action_ranking_failure"]
    assert tied_selection.qualifying is False
    assert "strict top-score" in tied_selection.reason
