"""Evaluate analytic baselines and a trained ActMask checkpoint on CPU."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from actmask.models import (
    ActMaskModel,
    ActionProximityMask,
    MotionMagnitudeMask,
    NoMask,
)
from actmask.training.train import build_dataset, seed_everything
from actmask.utils import BinaryMaskMetrics, load_config, resolve_project_path


def _batch_values(batch: Mapping[str, Any], key: str, count: int, default: Any) -> list[Any]:
    value = batch.get(key, default)
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().reshape(-1).tolist()
        return flattened if len(flattened) == count else [flattened] * count
    if isinstance(value, (list, tuple)):
        return list(value) if len(value) == count else [value] * count
    return [value] * count


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    *,
    return_records: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute global mask metrics and optionally retain per-sample records."""

    model = model.to(device)
    model.eval()
    accumulator = BinaryMaskMetrics(threshold=threshold, from_logits=True)
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device=device, dtype=torch.float32)
            velocities = batch["velocities"].to(device=device, dtype=torch.float32)
            actions = batch["action"].to(device=device, dtype=torch.float32)
            targets = batch["mask"].to(device=device, dtype=torch.float32)
            logits = model(points, velocities, actions)
            if logits.shape != targets.shape:
                raise RuntimeError(
                    f"Model returned {tuple(logits.shape)} for target {tuple(targets.shape)}"
                )
            accumulator.update(logits, targets)

            probabilities = torch.sigmoid(logits).detach().cpu()
            predictions = probabilities >= threshold
            target_masks = targets.detach().cpu() >= 0.5
            batch_size = int(targets.shape[0])
            pair_ids = _batch_values(batch, "pair_id", batch_size, -1)
            variant_ids = _batch_values(batch, "variant_id", batch_size, 0)
            scenarios = _batch_values(batch, "scenario", batch_size, "unknown")
            successes = _batch_values(batch, "success", batch_size, 0.0)

            for index in range(batch_size):
                records.append(
                    {
                        "pair_id": pair_ids[index],
                        "variant_id": variant_ids[index],
                        "scenario": str(scenarios[index]),
                        "success": float(successes[index]),
                        "score": float(probabilities[index].mean().item()),
                        "prediction": predictions[index].clone(),
                        "target": target_masks[index].clone(),
                    }
                )

    metrics = accumulator.compute()
    metrics["paired_scene_consistency"] = paired_scene_consistency(records)
    metrics["success_ranking_accuracy"] = success_ranking_accuracy(records)
    metrics["paired_comparisons"] = _paired_comparison_count(records)
    metrics["ranking_comparisons"] = _ranking_comparison_count(records)
    if return_records:
        return metrics, records
    return metrics


def _group_records(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, Any], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("scenario", "unknown")), record.get("pair_id", -1))].append(
            record
        )
    return grouped


def _record_pairs(
    records: Iterable[Mapping[str, Any]], *, differing_success: bool = False
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for group in _group_records(records).values():
        ordered = sorted(group, key=lambda item: int(item.get("variant_id", 0)))
        for first, second in itertools.combinations(ordered, 2):
            if differing_success and bool(first.get("success", 0.0)) == bool(
                second.get("success", 0.0)
            ):
                continue
            pairs.append((first, second))
    return pairs


def _paired_comparison_count(records: Iterable[Mapping[str, Any]]) -> int:
    return len(_record_pairs(records))


def paired_scene_consistency(records: Iterable[Mapping[str, Any]]) -> float:
    """Measure whether predicted changes match GT changes across paired variants.

    For each pair, both variants are XORed to form predicted and ground-truth
    change masks. Their IoU is one when both change masks are identically empty.
    """

    scores: list[float] = []
    for first, second in _record_pairs(records):
        first_prediction = torch.as_tensor(first["prediction"], dtype=torch.bool)
        second_prediction = torch.as_tensor(second["prediction"], dtype=torch.bool)
        first_target = torch.as_tensor(first["target"], dtype=torch.bool)
        second_target = torch.as_tensor(second["target"], dtype=torch.bool)
        predicted_change = torch.logical_xor(first_prediction, second_prediction)
        target_change = torch.logical_xor(first_target, second_target)
        intersection = int(torch.logical_and(predicted_change, target_change).sum().item())
        union = int(torch.logical_or(predicted_change, target_change).sum().item())
        scores.append(float(intersection / union) if union else 1.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _ranking_comparison_count(records: Iterable[Mapping[str, Any]]) -> int:
    return len(_record_pairs(records, differing_success=True))


def success_ranking_accuracy(records: Iterable[Mapping[str, Any]]) -> float:
    """Rank successful variants above unsuccessful variants; score ties as 0.5."""

    scores: list[float] = []
    for first, second in _record_pairs(records, differing_success=True):
        successful, unsuccessful = (
            (first, second) if bool(first.get("success", 0.0)) else (second, first)
        )
        score_delta = float(successful["score"]) - float(unsuccessful["score"])
        if abs(score_delta) <= 1.0e-12:
            scores.append(0.5)
        else:
            scores.append(1.0 if score_delta > 0.0 else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a mapping checkpoint at {path}")
    return checkpoint


def _checkpoint_path(config: Mapping[str, Any]) -> Path:
    evaluation_config = dict(config.get("evaluation", {}))
    configured_path = evaluation_config.get("checkpoint")
    if configured_path:
        return resolve_project_path(configured_path)
    training_config = dict(config.get("training", {}))
    output_dir = resolve_project_path(
        training_config.get("output_dir", config.get("output_dir", "outputs/actmask"))
    )
    return output_dir / str(training_config.get("checkpoint_name", "best_model.pt"))


def _evaluation_loader(config: Mapping[str, Any], split: str) -> DataLoader[Any]:
    dataset = build_dataset(config, split)
    evaluation_config = dict(config.get("evaluation", {}))
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    return DataLoader(
        dataset,
        batch_size=int(
            evaluation_config.get(
                "batch_size", dict(config.get("training", {})).get("batch_size", 8)
            )
        ),
        shuffle=False,
        num_workers=int(evaluation_config.get("num_workers", 0)),
        generator=generator,
        drop_last=False,
    )


def _print_table(results: Mapping[str, Mapping[str, Any]]) -> None:
    columns = (
        ("Model", 22),
        ("Precision", 9),
        ("Recall", 9),
        ("F1", 9),
        ("IoU", 9),
        ("PairCons", 9),
        ("SuccRank", 9),
    )
    header = " | ".join(label.ljust(width) for label, width in columns)
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        values = (
            model_name.ljust(22),
            f"{metrics['precision']:.3f}".rjust(9),
            f"{metrics['recall']:.3f}".rjust(9),
            f"{metrics['f1']:.3f}".rjust(9),
            f"{metrics['iou']:.3f}".rjust(9),
            f"{metrics['paired_scene_consistency']:.3f}".rjust(9),
            f"{metrics['success_ranking_accuracy']:.3f}".rjust(9),
        )
        print(" | ".join(values))


def run_evaluation(config: Mapping[str, Any], split: str | None = None) -> dict[str, Any]:
    """Evaluate all requested models and save a JSON report."""

    config = dict(config)
    seed_everything(int(config.get("seed", 0)))
    evaluation_config = dict(config.get("evaluation", {}))
    split = split or str(evaluation_config.get("split", "test"))
    threshold = float(evaluation_config.get("threshold", 0.5))
    loader = _evaluation_loader(config, split)

    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Learned ActMask checkpoint not found at {checkpoint_path}. "
            "Run python -m actmask.training.train first."
        )
    checkpoint = _load_checkpoint(checkpoint_path)
    model_config = dict(checkpoint.get("model_config", config.get("model", {})))
    learned_model = ActMaskModel(**model_config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    learned_model.load_state_dict(state_dict)

    models: dict[str, nn.Module] = {
        "NoMask": NoMask(),
        "MotionMagnitudeMask": MotionMagnitudeMask(),
        "ActionProximityMask": ActionProximityMask(),
        "ActMask": learned_model,
    }
    model_results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        metrics = evaluate_model(model, loader, device="cpu", threshold=threshold)
        assert isinstance(metrics, dict)
        model_results[name] = metrics

    _print_table(model_results)
    output_dir = resolve_project_path(
        evaluation_config.get(
            "output_dir",
            dict(config.get("training", {})).get(
                "output_dir", config.get("output_dir", "outputs/actmask")
            ),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / str(
        evaluation_config.get("metrics_name", "evaluation_metrics.json")
    )
    report: dict[str, Any] = {
        "split": split,
        "num_samples": len(loader.dataset),
        "threshold": threshold,
        "checkpoint": str(checkpoint_path),
        "models": model_results,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Evaluation metrics: {metrics_path}")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ActMask and CPU baselines")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--checkpoint", help="Override evaluation.checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Dataset split")
    parser.add_argument("--batch-size", type=int, help="Override evaluation.batch_size")
    parser.add_argument("--output-dir", help="Override evaluation.output_dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = load_config(args.config)
    evaluation_config = config.setdefault("evaluation", {})
    if args.checkpoint is not None:
        evaluation_config["checkpoint"] = args.checkpoint
    if args.batch_size is not None:
        evaluation_config["batch_size"] = args.batch_size
    if args.output_dir is not None:
        evaluation_config["output_dir"] = args.output_dir
    return run_evaluation(config, split=args.split)


if __name__ == "__main__":
    main()
