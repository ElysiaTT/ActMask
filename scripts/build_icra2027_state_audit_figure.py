#!/usr/bin/env python3
"""Build a compact, source-checked figure of the state audit construction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper_icra2027/figures/state_audit_construction.png"
SOURCES = {
    "outputs/actmask/milestone3r_nl_v2/full_run_config.json":
        "12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47",
    "actmask/experiments/milestone3r_nl_v2.py":
        "fcfc083a079b6d55b081f1fcd1bc46a67c123be28d798da60637f55f3ae0dcc2",
    "actmask/data/maniskill_3r_nl_tasks.py":
        "64f419586346e232e43b63abd2ea92ecbdde367d766758d552217059d6907a62",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sources() -> list[dict[str, str | int]]:
    inventory = []
    for relative, expected in SOURCES.items():
        path = ROOT / relative
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"state-figure source hash mismatch: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return inventory


def _audit_ladder(axis: plt.Axes) -> None:
    axis.set_axis_off()
    labels = (
        "Ordinary\ndata",
        "Match current\nstate + action",
        "Match unordered\nhistory",
        "Match final\ntwo frames",
        "Require mixed\ncandidates",
        "Challenge strongest\nfair control",
    )
    colors = ("#f4b183", "#f6cf9b", "#f3e3a1", "#cfe5bf", "#a8d8d8", "#9dc3e6")
    centers = np.linspace(0.075, 0.925, len(labels))
    width, height = 0.125, 0.58
    for index, (center, label, color) in enumerate(zip(centers, labels, colors)):
        axis.add_patch(
            Rectangle(
                (center - width / 2, 0.20),
                width,
                height,
                transform=axis.transAxes,
                facecolor=color,
                edgecolor="#24465f",
                linewidth=1.4,
            )
        )
        axis.text(center, 0.49, label, transform=axis.transAxes, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (center + width / 2 + 0.008, 0.49),
                    (centers[index + 1] - width / 2 - 0.008, 0.49),
                    transform=axis.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="#333333",
                )
            )
    axis.text(0.0, 0.97, "(a) Progressive shortcut removal and admission", transform=axis.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")


def _matched_history(axis: plt.Axes) -> None:
    frames = np.arange(1, 7)
    scale, ratio = 1.0, 0.30
    branch_a = np.asarray([-scale, -ratio * scale, ratio * scale, scale, 0.0, 0.0])
    branch_b = np.asarray([scale, ratio * scale, -ratio * scale, -scale, 0.0, 0.0])
    axis.axvspan(0.5, 4.5, color="#f4b183", alpha=0.22, linewidth=0)
    axis.axvspan(4.5, 6.5, color="#9dc3e6", alpha=0.28, linewidth=0)
    axis.plot(frames, branch_a, "o-", color="#c44e52", linewidth=2.1, markersize=5, label="history A")
    axis.plot(frames, branch_b, "s--", color="#4c72b0", linewidth=2.0, markersize=4.5, label="history B = early reverse")
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set_xlim(0.7, 6.3)
    axis.set_ylim(-1.25, 1.25)
    axis.set_xticks(frames)
    axis.set_xlabel("observation frame")
    axis.set_ylabel("signed target offset")
    axis.set_title("(b) Exact matched history pair", loc="left", fontsize=12, fontweight="bold")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    axis.text(3.9, -1.14, "early order differs", ha="right", va="bottom", fontsize=8.5, color="#8b4a20")
    axis.text(5.5, -1.14, "shared suffix", ha="center", va="bottom", fontsize=8.5, color="#24567a")
    axis.spines[["top", "right"]].set_visible(False)


def _branch_response(axis: plt.Axes) -> None:
    step = np.arange(0, 21, dtype=np.float64)
    damping, frequency = 0.15, 1.30
    response_a = 0.025 * step + 0.020 * (1.0 - np.exp(-damping * step)) * np.sin(frequency * step)
    response_b = -0.025 * step + 0.020 * (1.0 - np.exp(-damping * step)) * np.sin(frequency * step)
    candidate = 0.025 * step
    axis.fill_between(step[6:], candidate[6:] - 0.055, candidate[6:] + 0.055, color="#55a868", alpha=0.18, label="success radius")
    axis.plot(step, candidate, color="#222222", linestyle=":", linewidth=2.1, label="same + candidate")
    axis.plot(step, response_a, color="#c44e52", linewidth=2.1, label="installed response A")
    axis.plot(step, response_b, color="#4c72b0", linewidth=2.1, label="installed response B")
    axis.axvline(6, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(0, 20)
    axis.set_xlabel("candidate execution step")
    axis.set_ylabel("signed displacement")
    axis.set_title("(c) Same candidate, branch-specific outcome", loc="left", fontsize=12, fontweight="bold")
    axis.legend(frameon=False, fontsize=8.1, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)


def _task_maps(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("(d) Three task maps", loc="left", fontsize=12, fontweight="bold")

    # Damped moving capture.
    axis.add_patch(Rectangle((0.04, 0.15), 0.18, 0.18, facecolor="#dddddd", edgecolor="#555555"))
    axis.add_patch(Circle((0.13, 0.56), 0.055, facecolor="#c44e52", edgecolor="none"))
    axis.add_patch(FancyArrowPatch((0.13, 0.39), (0.13, 0.78), arrowstyle="<->", mutation_scale=13, color="#4c72b0", linewidth=1.7))
    axis.text(0.13, 0.07, "capture\n$y$ translation", ha="center", va="top", fontsize=9)

    # Elevated moving container.
    axis.add_patch(Rectangle((0.38, 0.16), 0.24, 0.10, facecolor="none", edgecolor="#555555", linewidth=1.5))
    axis.add_patch(Circle((0.50, 0.55), 0.052, facecolor="#c44e52", edgecolor="none"))
    axis.add_patch(FancyArrowPatch((0.35, 0.55), (0.65, 0.55), arrowstyle="<->", mutation_scale=13, color="#4c72b0", linewidth=1.7))
    axis.plot([0.50, 0.50], [0.28, 0.49], color="#777777", linestyle=":")
    axis.text(0.50, 0.07, "container\nelevated $x$", ha="center", va="top", fontsize=9)

    # Rotating slot.
    center = np.asarray([0.80, 0.50])
    axis.add_patch(Circle(center, 0.17, facecolor="none", edgecolor="#777777", linestyle=":"))
    angles = np.linspace(-0.75, 0.75, 30)
    axis.plot(center[0] + 0.17 * np.cos(angles), center[1] + 0.17 * np.sin(angles), color="#4c72b0", linewidth=2.0)
    axis.add_patch(Circle((center[0] + 0.17, center[1]), 0.045, facecolor="#c44e52", edgecolor="none"))
    axis.add_patch(FancyArrowPatch((0.74, 0.64), (0.85, 0.66), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=12, color="#4c72b0", linewidth=1.4))
    axis.text(0.80, 0.07, "rotating slot\nradius $0.13$", ha="center", va="top", fontsize=9)


def build(output: Path) -> dict[str, object]:
    sources = _check_sources()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    figure = plt.figure(figsize=(14.0, 5.8), dpi=160, facecolor="white")
    grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 2.25), hspace=0.38, wspace=0.30)
    _audit_ladder(figure.add_subplot(grid[0, :]))
    _matched_history(figure.add_subplot(grid[1, 0]))
    _branch_response(figure.add_subplot(grid[1, 1]))
    _task_maps(figure.add_subplot(grid[1, 2]))
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.96)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white", metadata={"Software": "ActMask"})
    plt.close(figure)

    from PIL import Image

    with Image.open(output) as image:
        width, height = image.size
    manifest = {
        "schema": "actmask-icra2027-state-audit-figure-v1",
        "sources": sources,
        "figure": {
            "filename": output.name,
            "width": width,
            "height": height,
            "sha256": _sha256(output),
        },
        "pass": True,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    manifest = build(arguments.output.resolve())
    print(json.dumps({"pass": True, "figure": manifest["figure"]}, indent=2))


if __name__ == "__main__":
    main()
