"""Save headless 3D comparisons of toy ActMask predictions.

Run from the project root with::

    python -m actmask.visualization.visualize_masks \
        --config configs/actmask/toy_cpu.yaml
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# This must happen before importing pyplot. The entrypoint never opens a window
# and is safe on a display-less CPU worker.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from actmask.data import ToyDynamicDataset
from actmask.models import (
    ActMaskModel,
    ActionProximityMask,
    MotionMagnitudeMask,
    NoMask,
)
from actmask.utils import load_config, resolve_project_path


def _dataset_from_config(config: Mapping[str, Any]) -> ToyDynamicDataset:
    data_config = dict(config.get("dataset", {}))
    visualization_config = dict(config.get("visualization", {}))
    split = str(visualization_config.get("split", "test"))
    seed_offsets = {"train": 0, "val": 1, "test": 2}
    if split not in seed_offsets:
        raise ValueError(f"visualization.split must be train, val, or test; got {split!r}")
    seed = int(
        data_config.get(f"{split}_seed", int(config.get("seed", 0)) + seed_offsets[split])
    )
    return ToyDynamicDataset(
        num_samples=int(
            data_config.get(f"{split}_samples", data_config.get("num_samples", 32))
        ),
        num_points=int(data_config.get("num_points", 128)),
        seed=seed,
        trajectory_steps=int(data_config.get("trajectory_steps", 16)),
    )


def _checkpoint_path(config: Mapping[str, Any]) -> Path:
    visualization_config = dict(config.get("visualization", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    training_config = dict(config.get("training", {}))

    explicit = visualization_config.get("checkpoint", evaluation_config.get("checkpoint"))
    if explicit:
        return resolve_project_path(str(explicit))

    output_dir = resolve_project_path(
        training_config.get("output_dir", config.get("output_dir", "outputs/actmask"))
    )
    return output_dir / str(training_config.get("checkpoint_name", "best_model.pt"))


def _load_checkpoint_cpu(path: Path) -> Mapping[str, Any]:
    """Load a local checkpoint onto CPU using the restricted loader when available."""

    try:
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:  # PyTorch versions predating the ``weights_only`` argument.
        payload = torch.load(path, map_location=torch.device("cpu"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping checkpoint at {path}, got {type(payload).__name__}")
    return payload


def _build_learned_model(
    config: Mapping[str, Any], checkpoint_path: Path
) -> tuple[ActMaskModel, bool]:
    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path.is_file():
        checkpoint = _load_checkpoint_cpu(checkpoint_path)

    configured_model = dict(config.get("model", {}))
    checkpoint_model = dict(checkpoint.get("model_config", {})) if checkpoint else {}
    model = ActMaskModel(**(checkpoint_model or configured_model)).cpu()

    if checkpoint is None:
        warnings.warn(
            f"No learned checkpoint found at {checkpoint_path}; the ActMask panel uses "
            "deterministically initialized, untrained weights. Run training first.",
            stacklevel=2,
        )
        model.eval()
        return model, False

    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {checkpoint_path} has no model state mapping")
    model.load_state_dict(state)
    model.eval()
    return model, True


def _predict_scores(
    sample: Mapping[str, Any], learned_model: ActMaskModel
) -> dict[str, np.ndarray]:
    points = torch.as_tensor(sample["points"], dtype=torch.float32).unsqueeze(0).cpu()
    velocities = torch.as_tensor(sample["velocities"], dtype=torch.float32).unsqueeze(0).cpu()
    action = torch.as_tensor(sample["action"], dtype=torch.float32).unsqueeze(0).cpu()

    models = {
        "NoMask": NoMask().cpu(),
        "Motion magnitude": MotionMagnitudeMask().cpu(),
        "Action proximity": ActionProximityMask().cpu(),
        "ActMask": learned_model,
    }
    scores: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, model in models.items():
            model.eval()
            logits = model(points, velocities, action)
            expected_shape = (1, points.shape[1])
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f"{type(model).__name__} returned {tuple(logits.shape)}, "
                    f"expected {expected_shape}"
                )
            scores[name] = torch.sigmoid(logits[0]).cpu().numpy()
    return scores


def _set_equal_3d_limits(ax: Any, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    half_extent = max(float((high - low).max()) / 2.0, 0.1) * 1.12
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_cloud_panel(
    ax: Any,
    points: np.ndarray,
    velocities: np.ndarray,
    action: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    max_velocity_vectors: int,
    velocity_scale: float,
) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=14,
        alpha=0.9,
        depthshade=False,
    )

    count = min(max(max_velocity_vectors, 1), len(points))
    vector_indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    selected_points = points[vector_indices]
    selected_velocities = velocities[vector_indices] * velocity_scale
    ax.quiver(
        selected_points[:, 0],
        selected_points[:, 1],
        selected_points[:, 2],
        selected_velocities[:, 0],
        selected_velocities[:, 1],
        selected_velocities[:, 2],
        color="black",
        linewidth=0.65,
        alpha=0.62,
        arrow_length_ratio=0.22,
    )

    start, end = action[:3], action[3:6]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color="darkorange",
        linewidth=2.8,
        label="action trajectory",
    )
    ax.scatter(*start, c="limegreen", marker="^", s=55, edgecolors="black", linewidths=0.5)
    ax.scatter(*end, c="gold", marker="X", s=60, edgecolors="black", linewidths=0.5)
    _set_equal_3d_limits(ax, np.concatenate((points, start[None], end[None]), axis=0))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x", labelpad=-3)
    ax.set_ylabel("y", labelpad=-3)
    ax.set_zlabel("z", labelpad=-3)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=24, azim=-58)


def _scalar(sample: Mapping[str, Any], key: str) -> Any:
    value = sample[key]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def save_sample_figure(
    sample: Mapping[str, Any],
    learned_model: ActMaskModel,
    output_path: Path,
    *,
    checkpoint_loaded: bool,
    max_velocity_vectors: int = 24,
    velocity_scale: float = 0.65,
) -> Path:
    """Render ground truth, three baselines, and ActMask for one sample."""

    points = torch.as_tensor(sample["points"], dtype=torch.float32).cpu().numpy()
    velocities = torch.as_tensor(sample["velocities"], dtype=torch.float32).cpu().numpy()
    action = torch.as_tensor(sample["action"], dtype=torch.float32).cpu().numpy()
    ground_truth = torch.as_tensor(sample["mask"], dtype=torch.float32).cpu().numpy()
    scores = _predict_scores(sample, learned_model)

    fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, index + 1, projection="3d") for index in range(5)]
    info_axis = fig.add_subplot(2, 3, 6)

    panel_values = [("Ground truth", ground_truth), *scores.items()]
    for ax, (title, values) in zip(axes, panel_values, strict=True):
        if title == "ActMask" and not checkpoint_loaded:
            title = "ActMask (untrained)"
        _draw_cloud_panel(
            ax,
            points,
            velocities,
            action,
            values,
            title,
            max_velocity_vectors=max_velocity_vectors,
            velocity_scale=velocity_scale,
        )

    scenario = str(_scalar(sample, "scenario"))
    pair_id = int(_scalar(sample, "pair_id"))
    variant_id = int(_scalar(sample, "variant_id"))
    success = float(_scalar(sample, "success"))
    info_axis.axis("off")
    info_axis.text(
        0.02,
        0.96,
        "Sample metadata\n"
        f"scenario: {scenario}\n"
        f"pair / variant: {pair_id} / {variant_id}\n"
        f"success: {success:.0f}\n\n"
        "Action\n"
        f"duration: {action[6]:.3f} s\n"
        f"gripper radius: {action[7]:.3f}\n\n"
        "Overlays\n"
        "black arrows: point velocity\n"
        "orange line: candidate trajectory\n"
        "green triangle: action start\n"
        "yellow X: action end\n\n"
        "Color: mask probability / binary GT",
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        linespacing=1.35,
    )

    scalar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    scalar_mappable.set_array([])
    fig.colorbar(
        scalar_mappable,
        ax=axes,
        shrink=0.68,
        pad=0.02,
        label="mask value / probability",
    )
    fig.suptitle("Action-conditioned future point-mask comparison", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _sample_indices(config: Mapping[str, Any], dataset_size: int) -> list[int]:
    visualization_config = dict(config.get("visualization", {}))
    configured = visualization_config.get("sample_indices")
    if configured is None:
        configured = [visualization_config.get("sample_index", 0)]
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        raise ValueError("visualization.sample_indices must be a sequence of integers")

    indices = [int(index) for index in configured]
    if not indices:
        raise ValueError("visualization.sample_indices cannot be empty")
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise IndexError(f"Visualization sample indices out of range: {invalid}")
    return indices


def run_visualization(config: Mapping[str, Any]) -> list[Path]:
    """Generate configured comparison figures and return their paths."""

    config = dict(config)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = _dataset_from_config(config)
    checkpoint_path = _checkpoint_path(config)
    learned_model, checkpoint_loaded = _build_learned_model(config, checkpoint_path)

    visualization_config = dict(config.get("visualization", {}))
    output_dir = resolve_project_path(
        visualization_config.get("output_dir", config.get("output_dir", "outputs/actmask"))
    )
    max_vectors = int(visualization_config.get("max_velocity_vectors", 24))
    velocity_scale = float(visualization_config.get("velocity_scale", 0.65))

    paths: list[Path] = []
    indices = _sample_indices(config, len(dataset))
    configured_filename = Path(
        str(visualization_config.get("filename", "mask_comparison.png"))
    )
    if configured_filename.is_absolute():
        raise ValueError("visualization.filename must be relative to the output directory")
    if configured_filename.suffix.lower() != ".png":
        raise ValueError("visualization.filename must end in .png")

    for index in indices:
        sample = dataset[index]
        scenario = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(_scalar(sample, "scenario")))
        filename = configured_filename
        if len(indices) > 1:
            filename = filename.with_name(
                f"{filename.stem}_{index:03d}_{scenario}{filename.suffix}"
            )
        output_path = output_dir / filename
        paths.append(
            save_sample_figure(
                sample,
                learned_model,
                output_path,
                checkpoint_loaded=checkpoint_loaded,
                max_velocity_vectors=max_vectors,
                velocity_scale=velocity_scale,
            )
        )

    checkpoint_status = str(checkpoint_path) if checkpoint_loaded else "not found (untrained panel)"
    print(f"Learned checkpoint: {checkpoint_status}")
    for path in paths:
        print(f"Saved visualization: {path}")
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CPU ActMask and baseline masks")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Override visualization.sample_indices with one test sample index",
    )
    parser.add_argument("--output-dir", help="Override the project-relative output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = _parse_args(argv)
    config = load_config(args.config)
    visualization_config = config.setdefault("visualization", {})
    if args.sample_index is not None:
        visualization_config["sample_indices"] = [args.sample_index]
    if args.output_dir is not None:
        visualization_config["output_dir"] = args.output_dir
    return run_visualization(config)


if __name__ == "__main__":
    main()
