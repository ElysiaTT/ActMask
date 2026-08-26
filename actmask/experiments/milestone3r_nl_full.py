"""Fixed-recipe full evaluation for the versioned 3R-NL matched-pair data.

This module intentionally consumes only the observable NPZ tensors.  Metadata
is used solely for grouped splitting and reporting, never as model input.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.experiments.milestone3r_analytic import ESTIMATORS
from actmask.experiments.milestone3r_metrics import calibration_error, pair_order, paired_bootstrap, ranking, roc_auc
from actmask.models.milestone3r_temporal import (
    ActionMLP, GRUTemporal, MaskedHistoryMLP, OrderedTemporalMLP, StaticMLP,
    TemporalConv1D, UnorderedHistoryMLP,
)


FAIR_KEYS = ("history", "timestamps", "visibility", "observation_confidence", "candidate_actions", "nominal_action_timing", "tcp_state")
MODEL_FACTORIES = {
    "GRU": lambda frame, action, horizon: GRUTemporal(frame, action),
    "TemporalConv1D": lambda frame, action, horizon: TemporalConv1D(frame, action),
    "OrderedTemporalMLP": lambda frame, action, horizon: OrderedTemporalMLP(frame * horizon, action),
    "MaskedHistoryMLP": lambda frame, action, horizon: MaskedHistoryMLP(frame * horizon, action),
    "UnorderedHistoryMLP": lambda frame, action, horizon: UnorderedHistoryMLP(frame * horizon, action),
    "StaticMLP": lambda frame, action, horizon: StaticMLP(frame - 3, action),
    "ActionMLP": lambda frame, action, horizon: ActionMLP(action),
}


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _indices(rows, split):
    return np.asarray([i for i, row in enumerate(rows) if row["split"] == split])


def _batch(value, device):
    return {key: torch.from_numpy(value[key]).float().to(device) for key in FAIR_KEYS if key != "nominal_action_timing"}


def _take(batch, indices):
    index = torch.from_numpy(indices).to(next(iter(batch.values())).device)
    return {key: value[index] for key, value in batch.items()}


def _velocity_score(velocity, actions):
    return (velocity[:, -3:] * actions.sum(1)).sum(1)


def _pair_correct(labels, scores, pairs):
    result = {}
    for group in np.unique(pairs):
        ix = np.flatnonzero(pairs == group)
        if len(ix) == 2 and labels[ix].sum() == 1:
            pos = ix[labels[ix].astype(bool)][0]
            neg = ix[~labels[ix].astype(bool)][0]
            result[str(group)] = float(scores[pos] > scores[neg]) + .5 * float(np.isclose(scores[pos], scores[neg]))
    return result


def _metric(rows, labels, scores, indices, candidates=20):
    selected = np.asarray(indices)
    candidates_mask = np.asarray([rows[i]["candidate_id"] < candidates for i in selected])
    selected = selected[candidates_mask]
    y, s = labels[selected], scores[selected]
    pairs = np.asarray([rows[i]["signed_group"] for i in selected])
    worlds = np.asarray([f"{rows[i]['family']}:{rows[i]['mechanism']}:{rows[i]['world_id']}" for i in selected])
    result = ranking(y, s, worlds)
    result.update(pair_order_accuracy=pair_order(y, s, pairs), ranking_roc_auc=roc_auc(y, s), calibration_ece=calibration_error(y, s), examples=int(len(selected)), candidate_prefix=int(candidates))
    return result


def _fit(factory, batch, train, labels, seed, steps):
    torch.manual_seed(seed)
    model = factory().to(batch["history"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015)
    target = torch.from_numpy(labels[train]).float().to(batch["history"].device)
    train_batch = _take(batch, train)
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(model(train_batch), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model.eval()


def _corrupt(batch, name):
    output = {key: value.clone() for key, value in batch.items()}
    history = output["history"]
    if name == "correct_ordered":
        return output
    if name == "exact_reversed":
        for key in ("history", "timestamps", "visibility", "observation_confidence"):
            output[key] = torch.flip(output[key], (1,))
    elif name == "last_two_only" or name == "earlier_zeroed":
        output["history"][:, :-2] = output["history"][:, -1:].expand(-1, history.shape[1] - 2, -1) if name == "last_two_only" else 0
    elif name == "independent_permutation":
        generator = torch.Generator(device=history.device); generator.manual_seed(731)
        perm = torch.rand((history.shape[0], history.shape[1]), generator=generator, device=history.device).argsort(1)
        for key in ("history", "timestamps", "visibility", "observation_confidence"):
            value = output[key]; expand = perm[(...,) + (None,) * (value.ndim - 2)].expand_as(value)
            output[key] = value.gather(1, expand)
    elif name == "unordered_multiset":
        output["history"] = torch.sort(history, dim=1).values
    elif name == "timestamps_removed":
        output["timestamps"].zero_()
    elif name == "timestamps_shuffled":
        output["timestamps"] = torch.flip(output["timestamps"], (1,))
    elif name == "history_mismatched":
        generator = torch.Generator(device=history.device); generator.manual_seed(193)
        perm = torch.randperm(history.shape[0], generator=generator, device=history.device)
        for key in ("history", "timestamps", "visibility", "observation_confidence"):
            output[key] = output[key][perm]
    else:
        raise ValueError(name)
    return output


def derive_candidate_prefixes(source, destination, counts=(5, 10)):
    """Materialize C5/C10 aligned prefixes without another simulator run."""
    source, destination = Path(source), Path(destination)
    rows = _rows(source / "metadata.jsonl")
    values = load_model_inputs(source / "model_inputs.npz")
    labels = np.load(source / "labels.npz")
    outputs = {}
    for count in counts:
        keep = np.asarray([row["candidate_id"] < count for row in rows])
        target = destination / f"full_c{count}"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "model_inputs.npz", **{key: value[keep] for key, value in values.items()})
        np.savez_compressed(target / "labels.npz", **{key: value[keep] for key, value in labels.items()})
        (target / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row, flag in zip(rows, keep) if flag))
        report = {"source": str(source), "candidate_prefix": count, "examples": int(keep.sum()), "groups": int(keep.sum() // 2), "simulator_executions_added": 0}
        (target / "derivation_report.json").write_text(json.dumps(report, indent=2))
        outputs[f"C{count}"] = report
    return outputs


def run(root, output, *, diagnostic_seeds=(17, 29, 43), final_seeds=(17, 29, 43, 59, 71), steps=120):
    root, output = Path(root), Path(output)
    value = load_model_inputs(root / "model_inputs.npz")
    if set(value) != set(FAIR_KEYS):
        raise AssertionError(f"input whitelist violation: {set(value)}")
    labels = np.load(root / "labels.npz")["success"].astype(np.float32)
    rows = _rows(root / "metadata.jsonl")
    train, validation, test = (_indices(rows, name) for name in ("train", "val", "test"))
    device = torch.device("cuda:0")
    batch = _batch(value, device)
    pairs = np.asarray([row["signed_group"] for row in rows])
    frame = value["history"].shape[2] + 3
    action = value["candidate_actions"].shape[1] * 3 + 3
    horizon = value["history"].shape[1]
    records = []
    analytic_scores = {}
    for name, estimator in ESTIMATORS.items():
        velocity = estimator(value["history"], value["timestamps"], value["visibility"])
        score = _velocity_score(velocity, value["candidate_actions"])
        analytic_scores[name] = score
        for split, indices in (("validation", validation), ("test", test)):
            records.append(dict(method=name, kind="analytic", seed=None, split=split, **_metric(rows, labels, score, indices)))
    # The oracle exposes branch metadata and is explicitly diagnostic-only.
    oracle = np.asarray([row["branch"] for row in rows], dtype=np.float32)
    for split, indices in (("validation", validation), ("test", test)):
        records.append(dict(method="OracleBranchDiagnostic", kind="oracle_not_fair", seed=None, split=split, **_metric(rows, labels, oracle, indices)))

    learned_scores = {}
    for name, build in MODEL_FACTORIES.items():
        for seed in diagnostic_seeds:
            model = _fit(lambda: build(frame, action, horizon), batch, train, labels, seed, steps)
            with torch.no_grad():
                score = model(batch).cpu().numpy()
            learned_scores[(name, seed)] = score
            for split, indices in (("validation", validation), ("test", test)):
                records.append(dict(method=name, kind="learned", seed=seed, split=split, **_metric(rows, labels, score, indices)))
    validation_mean = {name: float(np.mean([r["pair_order_accuracy"] for r in records if r["kind"] == "learned" and r["method"] == name and r["split"] == "validation"])) for name in MODEL_FACTORIES}
    selected = max(("GRU", "TemporalConv1D", "OrderedTemporalMLP", "MaskedHistoryMLP"), key=validation_mean.get)
    # Fixed five-seed confirmation of the one validation-selected recipe.
    final_scores, final_models = [], []
    for seed in final_seeds:
        model = _fit(lambda: MODEL_FACTORIES[selected](frame, action, horizon), batch, train, labels, seed, steps)
        final_models.append(model)
        with torch.no_grad():
            score = model(batch).cpu().numpy()
        final_scores.append(score)
        for split, indices in (("validation", validation), ("test", test)):
            records.append(dict(method=f"{selected}FiveSeed", kind="learned_confirmation", seed=seed, split=split, **_metric(rows, labels, score, indices)))
    mean_final = np.mean(final_scores, axis=0)
    best_analytic = max(analytic_scores, key=lambda name: _metric(rows, labels, analytic_scores[name], validation)["pair_order_accuracy"])
    learned_pair = _pair_correct(labels[test], mean_final[test], pairs[test])
    analytic_pair = _pair_correct(labels[test], analytic_scores[best_analytic][test], pairs[test])
    shared = sorted(set(learned_pair) & set(analytic_pair))
    delta = np.asarray([learned_pair[g] - analytic_pair[g] for g in shared])
    ci = paired_bootstrap(delta, np.asarray(shared), seed=3101) if shared else {"delta": float("nan"), "ci95": [float("nan"), float("nan")], "groups": 0}

    causal = []
    for name in ("correct_ordered", "exact_reversed", "last_two_only", "earlier_zeroed", "independent_permutation", "unordered_multiset", "timestamps_removed", "timestamps_shuffled", "history_mismatched"):
        with torch.no_grad():
            score = np.mean([model(_corrupt(batch, name)).cpu().numpy() for model in final_models], axis=0)
        causal.append(dict(condition=name, **_metric(rows, labels, score, test)))
    causal_map = {item["condition"]: item for item in causal}

    # Reuse C20 rows as aligned C5/C10 prefixes; no second simulator pass.
    ranking_prefixes = {f"C{count}": _metric(rows, labels, mean_final, test, count) for count in (5, 10, 20)}
    timings = []
    for start in range(0, min(len(test), 800), 40):
        index = test[start:start + 40]
        torch.cuda.synchronize(); begun = time.perf_counter()
        with torch.no_grad():
            _ = final_models[0](_take(batch, index))
        torch.cuda.synchronize(); timings.append((time.perf_counter() - begun) * 1000)
    report = dict(
        protocol="fixed C20 GPU-PhysX full data; C5/C10 are aligned action-prefix views",
        input_whitelist=list(FAIR_KEYS), split_counts={"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        diagnostic_seeds=list(diagnostic_seeds), final_seeds=list(final_seeds), steps=steps,
        validation_selected_learned=selected, validation_pair_order=validation_mean,
        selected_analytic=best_analytic, learned_over_analytic_grouped_bootstrap=ci,
        records=records, ranking_prefixes=ranking_prefixes, causal_controls=causal,
        causal_diagnostics=dict(earlier_history_advantage=causal_map["correct_ordered"]["pair_order_accuracy"] - causal_map["last_two_only"]["pair_order_accuracy"], reversal_causal_drop=causal_map["correct_ordered"]["pair_order_accuracy"] - causal_map["exact_reversed"]["pair_order_accuracy"]),
        c20_latency_ms=dict(samples=len(timings), p95=float(np.percentile(timings, 95)), mean=float(np.mean(timings))),
        ood_status="not generated in the 100,000-execution first full run; no nonlinear-OOD pass claim is permitted",
        candidate_action_limitation="candidate pulses differ, but all C20 candidates retain the same branch outcome in this generated run; ranking is over strict counterfactual pair members, not demonstrated action-effect diversity.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_evaluation.json").write_text(json.dumps(report, indent=2))
    return report
