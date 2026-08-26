"""Minimal deterministic training entrypoint for ActMask.

Run from the project root with::

    python -m actmask.training.train --config configs/actmask/toy_cpu.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from actmask.data import ToyDynamicDataset
from actmask.models import ActMaskModel
from actmask.utils import BinaryMaskMetrics, load_config, resolve_project_path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_dataset(config: Mapping[str, Any], split: str) -> ToyDynamicDataset:
    """Construct one deterministic dataset split from the shared config."""

    data_config = dict(config.get("dataset", {}))
    split = str(split)
    default_offsets = {"train": 0, "val": 1, "test": 2}
    base_seed = int(config.get("seed", 0))
    num_samples = int(data_config.get(f"{split}_samples", data_config.get("num_samples", 32)))
    split_seed = int(
        data_config.get(f"{split}_seed", base_seed + default_offsets.get(split, 0))
    )
    return ToyDynamicDataset(
        num_samples=num_samples,
        num_points=int(data_config.get("num_points", 128)),
        seed=split_seed,
        trajectory_steps=int(data_config.get("trajectory_steps", 16)),
    )


def _build_loader(
    dataset: Dataset[Any],
    config: Mapping[str, Any],
    *,
    split: str,
    shuffle: bool,
) -> DataLoader[Any]:
    training_config = dict(config.get("training", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    section = training_config if split in {"train", "val"} else evaluation_config
    batch_size = int(section.get("batch_size", training_config.get("batch_size", 8)))
    num_workers = int(section.get("num_workers", 0))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 0)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def compute_positive_weight(dataset: Dataset[Any], max_weight: float | None = None) -> float:
    """Compute ``negative / positive`` for sparse-mask BCE weighting."""

    positives = 0.0
    total = 0
    for index in range(len(dataset)):
        mask = torch.as_tensor(dataset[index]["mask"], dtype=torch.float32)
        positives += float(mask.sum().item())
        total += mask.numel()

    negatives = float(total) - positives
    weight = negatives / positives if positives > 0.0 else 1.0
    if max_weight is not None:
        weight = min(weight, float(max_weight))
    return max(weight, 1.0e-8)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device | str,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metric_accumulator = BinaryMaskMetrics(threshold=threshold, from_logits=True)
    total_loss = 0.0
    total_points = 0

    for batch in loader:
        points = batch["points"].to(device=device, dtype=torch.float32)
        velocities = batch["velocities"].to(device=device, dtype=torch.float32)
        actions = batch["action"].to(device=device, dtype=torch.float32)
        targets = batch["mask"].to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(points, velocities, actions)
            if logits.shape != targets.shape:
                raise RuntimeError(
                    f"Model returned {tuple(logits.shape)} for target {tuple(targets.shape)}"
                )
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        point_count = targets.numel()
        total_loss += float(loss.detach().item()) * point_count
        total_points += point_count
        metric_accumulator.update(logits, targets)

    if total_points == 0:
        raise ValueError("Cannot run an epoch over an empty data loader")
    metrics = metric_accumulator.compute()
    metrics["loss"] = total_loss / total_points
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Train for one epoch and return loss plus global mask metrics."""

    return _run_epoch(model, loader, criterion, device, threshold, optimizer)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate one split without gradients."""

    return _run_epoch(model, loader, criterion, device, threshold, optimizer=None)


def _select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _build_optimizer(
    model: nn.Module, training_config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    learning_rate = float(training_config.get("learning_rate", 1.0e-3))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {optimizer_name!r}; use 'adam' or 'adamw'")


def _json_ready_metrics(metrics: Mapping[str, float | int]) -> dict[str, float | int]:
    return {key: value for key, value in metrics.items()}


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Train ActMask from a loaded config and persist the best checkpoint."""

    config = deepcopy(dict(config))
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    training_config = dict(config.get("training", {}))
    num_threads = int(training_config.get("num_threads", 1))
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    device = _select_device(str(config.get("device", "cpu")))
    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    train_loader = _build_loader(train_dataset, config, split="train", shuffle=True)
    val_loader = _build_loader(val_dataset, config, split="val", shuffle=False)

    model = ActMaskModel(**dict(config.get("model", {}))).to(device)
    optimizer = _build_optimizer(model, training_config)
    use_positive_weight = bool(training_config.get("compute_pos_weight", True))
    positive_weight = (
        compute_positive_weight(train_dataset, training_config.get("max_pos_weight"))
        if use_positive_weight
        else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )

    epochs = int(training_config.get("epochs", 5))
    threshold = float(training_config.get("threshold", 0.5))
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    output_dir = resolve_project_path(
        training_config.get("output_dir", config.get("output_dir", "outputs/actmask"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / str(training_config.get("checkpoint_name", "best_model.pt"))
    metrics_path = output_dir / str(training_config.get("metrics_name", "training_metrics.json"))

    history: list[dict[str, Any]] = []
    best_iou = -1.0
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device=device, threshold=threshold
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device=device, threshold=threshold
        )
        history.append(
            {
                "epoch": epoch,
                "train": _json_ready_metrics(train_metrics),
                "validation": _json_ready_metrics(val_metrics),
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"P/R/F1/IoU {train_metrics['precision']:.3f}/"
            f"{train_metrics['recall']:.3f}/{train_metrics['f1']:.3f}/"
            f"{train_metrics['iou']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"P/R/F1/IoU {val_metrics['precision']:.3f}/"
            f"{val_metrics['recall']:.3f}/{val_metrics['f1']:.3f}/"
            f"{val_metrics['iou']:.3f}"
        )

        current_iou = float(val_metrics["iou"])
        current_loss = float(val_metrics["loss"])
        is_better = current_iou > best_iou or (
            current_iou == best_iou and current_loss < best_loss
        )
        if is_better:
            best_iou = current_iou
            best_loss = current_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": _json_ready_metrics(val_metrics),
                    "model_config": dict(config.get("model", {})),
                    "seed": seed,
                },
                checkpoint_path,
            )

    result: dict[str, Any] = {
        "device": str(device),
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "best_validation_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Best checkpoint: {checkpoint_path} (epoch {best_epoch}, IoU {best_iou:.3f})")
    print(f"Training metrics: {metrics_path}")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CPU ActMask prototype")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate")
    parser.add_argument("--output-dir", help="Override the project-relative output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = load_config(args.config)
    training_config = config.setdefault("training", {})
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.output_dir is not None:
        training_config["output_dir"] = args.output_dir
    return run_training(config)


if __name__ == "__main__":
    main()
