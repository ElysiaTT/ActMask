#!/usr/bin/env python3
"""Frozen external transition-necessity audit for WAV MiniGrid.

The script expects a checkout of world-action-verifier/wav_minigrid at the
commit and file hashes frozen in docs/icra2027_wav_external_audit_prereg.md.
It never modifies the upstream checkout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


UPSTREAM_COMMIT = "527159b06149beacfb3b2d77af7d938ca4efa32d"
SEEDS = (9102, 9103, 9104, 9105, 9106)
COMPLEXITIES = (6, 8, 10, 12, 14)
EXPECTED_HASHES = {
    "checkpoints/pretrained_inverse_model_for_state_complexity.pth":
        "2d0ac5c684c90fc08ed4e280f8944b5d3f8f2be1c1ff8c990eeb708eb2bd67d1",
    "data/MiniGrid-Empty-Interact-6x6-o6-v0_train.npz":
        "38304df2e59effecd664880fa3e708ef0d3fcf3bd2986a63ca395efaf7c01378",
    "data/MiniGrid-Empty-Interact-6x6-o6-v0_test.npz":
        "288534d7575aeb0c95fe2301cc6db87806e7d48f27471b578e9b5b75dcb1b9c0",
    "data/MiniGrid-Empty-Interact-6x6-o8-v0_test.npz":
        "1a4974fde645cce28a1e4679ad3442934ccc23dbf2fb4eb81181fc00fcb7683c",
    "data/MiniGrid-Empty-Interact-6x6-o10-v0_test.npz":
        "ab4900f995101caf261c90d21435692714444c97dced3a21eee449263891370f",
    "data/MiniGrid-Empty-Interact-6x6-o12-v0_test.npz":
        "8602272e50412a7526f7c77b4ad961407ac3f92238271a0d0b86ba5776699ea3",
    "data/MiniGrid-Empty-Interact-6x6-o14-v0_test.npz":
        "7986f1fbb962b7d4cb0c7e6f5bda044856370946a35c4e376d69dca1df74f3a7",
    "src/wav_minigrid/models/idm.py":
        "7a7b4135d9528aed69f6d44228ead2ff6fa3d82cd8b78e782093051fd821bb72",
    "src/wav_minigrid/evaluate_generation.py":
        "9b0d950a8f8d2c8e7f556fca0747b8113de8bb7ed6c8a52f2eed00f6a8adbcc1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream(root: Path) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen upstream input: {path}")
        actual = sha256_file(path)
        observed[rel] = actual
        if actual != expected:
            raise ValueError(f"hash mismatch for {rel}: {actual} != {expected}")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit mismatch: {commit} != {UPSTREAM_COMMIT}")
    return {"commit": commit, "sha256": observed}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["states"] = data["states"].reshape(-1, 6, 6, 3).astype(np.int64)
    data["next_states"] = data["next_states"].reshape(-1, 6, 6, 3).astype(np.int64)
    data["carried"] = data["carried"].reshape(-1, 2).astype(np.int64)
    data["next_carried"] = data["next_carried"].reshape(-1, 2).astype(np.int64)
    data["actions"] = data["actions"].reshape(-1).astype(np.int64)
    return data


def stratified_train_val(actions: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for action in sorted(np.unique(actions).tolist()):
        indices = np.flatnonzero(actions == action)
        rng.shuffle(indices)
        n_val = max(1, int(round(0.2 * len(indices))))
        val_parts.append(indices[:n_val])
        train_parts.append(indices[n_val:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    return train.astype(np.int64), val.astype(np.int64)


def _frame_one_hot(frame: np.ndarray) -> np.ndarray:
    obj = np.eye(11, dtype=np.float32)[frame[..., 0]]
    color = np.eye(6, dtype=np.float32)[frame[..., 1]]
    state = np.eye(4, dtype=np.float32)[frame[..., 2]]
    return np.concatenate([obj, color, state], axis=-1).transpose(0, 3, 1, 2)


def _carried_one_hot(carried: np.ndarray) -> np.ndarray:
    color = np.eye(6, dtype=np.float32)[carried[:, 0]]
    obj = np.eye(11, dtype=np.float32)[carried[:, 1]]
    return np.concatenate([color, obj], axis=1)


def select_evidence(
    data: dict[str, np.ndarray], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "current":
        frames = (data["states"], data["states"])
        carried = (data["carried"], data["carried"])
    elif mode == "next":
        frames = (data["next_states"], data["next_states"])
        carried = (data["next_carried"], data["next_carried"])
    elif mode == "paired":
        frames = (data["states"], data["next_states"])
        carried = (data["carried"], data["next_carried"])
    else:
        raise ValueError(f"unknown evidence mode: {mode}")
    frame_x = np.concatenate([_frame_one_hot(x) for x in frames], axis=1)
    carried_x = np.concatenate([_carried_one_hot(x) for x in carried], axis=1)
    return frame_x, carried_x


class MatchedEvidenceCNN(nn.Module):
    """One architecture for current-only, next-only, and paired cells."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(42, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 6 * 6 + 34, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, frame_x: torch.Tensor, carried_x: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(frame_x).flatten(1)
        return self.head(torch.cat([encoded, carried_x], dim=1))


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accuracy(model: nn.Module, x: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor,
              indices: np.ndarray, device: torch.device, batch_size: int) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
            logits = model(x[0][idx].to(device), x[1][idx].to(device))
            correct += int((logits.argmax(1).cpu() == y[idx]).sum())
    return correct / max(1, len(indices))


def train_matched_model(
    data: dict[str, np.ndarray], mode: str, seed: int, device: torch.device,
    max_epochs: int, patience: int, batch_size: int, learning_rate: float,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    train_idx, val_idx = stratified_train_val(data["actions"], seed)
    frame_np, carried_np = select_evidence(data, mode)
    frame_x = torch.from_numpy(frame_np)
    carried_x = torch.from_numpy(carried_np)
    y = torch.from_numpy(data["actions"]).long()
    model = MatchedEvidenceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_gain = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            logits = model(frame_x[idx].to(device), carried_x[idx].to(device))
            loss = F.cross_entropy(logits, y[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_acc = _accuracy(
            model, (frame_x, carried_x), y, val_idx, device, batch_size
        )
        if val_acc > best_acc + 1e-12:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)),
        "selected_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "validation_action_accuracy": float(best_acc),
        "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
    }


def predict_matched(
    model: nn.Module, data: dict[str, np.ndarray], mode: str,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    frame_np, carried_np = select_evidence(data, mode)
    pred: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame_np), batch_size):
            frame = torch.from_numpy(frame_np[start:start + batch_size]).to(device)
            carried = torch.from_numpy(carried_np[start:start + batch_size]).to(device)
            pred.append(model(frame, carried).argmax(1).cpu().numpy())
    return np.concatenate(pred).astype(np.int64)


def predict_sparse_idm(
    model: nn.Module, data: dict[str, np.ndarray], seed: int,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    set_seed(seed)
    pred: list[np.ndarray] = []
    n = len(data["actions"])
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            frames = torch.from_numpy(np.stack([
                data["states"][start:end], data["next_states"][start:end]
            ], axis=0)).float().to(device)
            cols = torch.from_numpy(np.stack([
                data["carried"][start:end, 0], data["next_carried"][start:end, 0]
            ], axis=0)[..., None]).long().to(device)
            objs = torch.from_numpy(np.stack([
                data["carried"][start:end, 1], data["next_carried"][start:end, 1]
            ], axis=0)[..., None]).long().to(device)
            logits, _, _ = model({
                "frame": frames, "carried_col": cols, "carried_obj": objs
            })
            pred.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(pred).astype(np.int64)


DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _agent_pose(frame: np.ndarray) -> tuple[tuple[int, int], int]:
    where = np.argwhere(frame[..., 0] == 10)
    if len(where) != 1:
        raise ValueError(f"expected one agent, found {len(where)}")
    pos = (int(where[0, 0]), int(where[0, 1]))
    return pos, int(frame[pos][2]) % 4


def transition_rule_one(
    current: np.ndarray, nxt: np.ndarray,
    carried: np.ndarray, next_carried: np.ndarray,
) -> int:
    """Decode a physical action, allowing outcome-equivalent pickup/swap."""
    pos, direction = _agent_pose(current)
    next_pos, next_direction = _agent_pose(nxt)
    if pos != next_pos:
        return 2
    direction_delta = (next_direction - direction) % 4
    if direction_delta == 3:
        return 0
    if direction_delta == 1:
        return 1
    dy, dx = DIR_TO_VEC[direction]
    front = (pos[0] + dy, pos[1] + dx)
    carried_changed = not np.array_equal(carried, next_carried)
    if carried_changed:
        before_empty = int(carried[1]) == 1
        after_empty = int(next_carried[1]) == 1
        if before_empty and not after_empty:
            return 3  # pickup and empty-hand swap are outcome-equivalent
        if not before_empty and not after_empty:
            return 6
        if not before_empty and after_empty:
            if np.array_equal(current[front], nxt[front]):
                return 5  # toggling a box clears carried state in this oracle
            return 4
    if not np.array_equal(current[front], nxt[front]):
        return 5
    return 6


def predict_transition_rule(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([
        transition_rule_one(s, n, c, nc)
        for s, n, c, nc in zip(
            data["states"], data["next_states"],
            data["carried"], data["next_carried"],
        )
    ], dtype=np.int64)


def confusion_matrix(target: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (target, pred), 1)
    return matrix.tolist()


def dynamic_accuracy_counts(
    data: dict[str, np.ndarray], pred: np.ndarray, oracle: Any,
) -> tuple[int, int]:
    numerator = 0
    denominator = 0
    for current, nxt, carried, next_carried, action in zip(
        data["states"], data["next_states"], data["carried"],
        data["next_carried"], pred,
    ):
        simulated, sim_col, sim_obj = oracle.step(
            current.copy(), int(carried[0]), int(carried[1]), int(action)
        )
        raw_diff = np.abs(current.astype(np.float64) - nxt.astype(np.float64)).sum(axis=-1) > 0.01
        noise_floor = (current[..., 0] == 3) & (nxt[..., 0] == 3)
        dynamic_mask = raw_diff & ~noise_floor
        cell_correct = np.all(simulated.astype(np.int64) == nxt, axis=-1)
        numerator += int(np.sum(cell_correct & dynamic_mask))
        denominator += int(np.sum(dynamic_mask))
        carried_changed = not np.array_equal(carried, next_carried)
        if carried_changed:
            numerator += int(int(sim_col) == int(next_carried[0]) and int(sim_obj) == int(next_carried[1]))
            denominator += 1
    return numerator, denominator


def score_predictions(
    data: dict[str, np.ndarray], pred: np.ndarray, oracle: Any,
) -> dict[str, Any]:
    numerator, denominator = dynamic_accuracy_counts(data, pred, oracle)
    return {
        "count": int(len(pred)),
        "action_correct": int(np.sum(pred == data["actions"])),
        "action_accuracy": float(np.mean(pred == data["actions"])),
        "dynamic_correct": int(numerator),
        "dynamic_total": int(denominator),
        "dynamic_accuracy": float(numerator / max(1, denominator)),
        "confusion": confusion_matrix(data["actions"], pred),
        "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
    }


def summarize_method(per_seed: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_complexity": {}}
    for complexity in COMPLEXITIES:
        cells = [entry["tests"][str(complexity)] for entry in per_seed.values()]
        summary["by_complexity"][str(complexity)] = {}
        for metric in ("action_accuracy", "dynamic_accuracy"):
            values = np.asarray([cell[metric] for cell in cells], dtype=np.float64)
            summary["by_complexity"][str(complexity)][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    for metric in ("action_accuracy", "dynamic_accuracy"):
        seed_macro = np.asarray([
            np.mean([entry["tests"][str(c)][metric] for c in COMPLEXITIES])
            for entry in per_seed.values()
        ], dtype=np.float64)
        summary[f"macro_{metric}"] = {
            "mean": float(seed_macro.mean()),
            "sample_std": float(seed_macro.std(ddof=1)) if len(seed_macro) > 1 else 0.0,
        }
    return summary


def evaluate_decision(summaries: dict[str, Any]) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    for complexity in COMPLEXITIES:
        key = str(complexity)
        paired = summaries["paired"]["by_complexity"][key]["dynamic_accuracy"]["mean"]
        current = summaries["current_only"]["by_complexity"][key]["dynamic_accuracy"]["mean"]
        nxt = summaries["next_only"]["by_complexity"][key]["dynamic_accuracy"]["mean"]
        sparse = summaries["sparse_idm"]["by_complexity"][key]["dynamic_accuracy"]["mean"]
        rule = summaries["transition_rule"]["by_complexity"][key]["dynamic_accuracy"]["mean"]
        strongest_single = max(current, nxt)
        gates.extend([
            {"complexity": complexity, "gate": "paired_margin", "value": paired - strongest_single,
             "threshold": 0.10, "pass": paired - strongest_single >= 0.10 - 1e-12},
            {"complexity": complexity, "gate": "sparse_idm_floor", "value": sparse,
             "threshold": 0.90, "pass": sparse >= 0.90 - 1e-12},
            {"complexity": complexity, "gate": "rule_beats_single", "value": rule - strongest_single,
             "threshold": 0.0, "strict": True, "pass": rule > strongest_single + 1e-12},
        ])
    return {"pass": all(item["pass"] for item in gates), "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/actmask/icra2027_wav_external_audit/audit.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if (args.max_epochs, args.patience, args.batch_size, args.learning_rate) != (200, 25, 64, 1e-3):
        raise ValueError("frozen training hyperparameters may not be overridden")

    started = time.time()
    wav_root = args.wav_root.resolve()
    provenance = verify_upstream(wav_root)
    sys.path.insert(0, str(wav_root / "src"))
    from wav_minigrid.evaluate_generation import MiniGridPhysicsOracle
    from wav_minigrid.models import SparseIDM

    train_data = load_npz(wav_root / "data/MiniGrid-Empty-Interact-6x6-o6-v0_train.npz")
    tests = {
        complexity: load_npz(
            wav_root / f"data/MiniGrid-Empty-Interact-6x6-o{complexity}-v0_test.npz"
        ) for complexity in COMPLEXITIES
    }
    device = torch.device(args.device)
    oracle = MiniGridPhysicsOracle()
    methods: dict[str, dict[str, Any]] = {
        "train_prior": {}, "current_only": {}, "next_only": {}, "paired": {},
        "sparse_idm": {}, "transition_rule": {},
    }

    for seed in SEEDS:
        train_idx, _ = stratified_train_val(train_data["actions"], seed)
        counts = np.bincount(train_data["actions"][train_idx], minlength=7)
        prior_action = int(np.flatnonzero(counts == counts.max())[0])
        methods["train_prior"][str(seed)] = {
            "selected_action": prior_action,
            "tests": {
                str(c): score_predictions(d, np.full(len(d["actions"]), prior_action), oracle)
                for c, d in tests.items()
            },
        }
        for mode, name in (
            ("current", "current_only"), ("next", "next_only"), ("paired", "paired")
        ):
            model, train_report = train_matched_model(
                train_data, mode, seed, device, args.max_epochs, args.patience,
                args.batch_size, args.learning_rate,
            )
            methods[name][str(seed)] = {
                "training": train_report,
                "tests": {
                    str(c): score_predictions(
                        d, predict_matched(model, d, mode, device, args.batch_size), oracle
                    ) for c, d in tests.items()
                },
            }
            print(
                f"seed={seed} mode={mode} val={train_report['validation_action_accuracy']:.4f} "
                f"epoch={train_report['selected_epoch']}", flush=True,
            )

    sparse = SparseIDM(grid_h=6, grid_w=6, num_actions=7).to(device)
    checkpoint = torch.load(
        wav_root / "checkpoints/pretrained_inverse_model_for_state_complexity.pth",
        map_location=device, weights_only=True,
    )
    incompatible = sparse.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"SparseIDM checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    sparse.eval()
    for seed in SEEDS:
        methods["sparse_idm"][str(seed)] = {
            "tests": {
                str(c): score_predictions(
                    d, predict_sparse_idm(sparse, d, seed, device, args.batch_size), oracle
                ) for c, d in tests.items()
            }
        }

    methods["transition_rule"]["fixed"] = {
        "tests": {
            str(c): score_predictions(d, predict_transition_rule(d), oracle)
            for c, d in tests.items()
        }
    }
    summaries = {name: summarize_method(per_seed) for name, per_seed in methods.items()}
    decision = evaluate_decision(summaries)
    protocol_path = Path(__file__).resolve().parents[1] / "docs/icra2027_wav_external_audit_prereg.md"
    report = {
        "schema_version": 1,
        "protocol": "docs/icra2027_wav_external_audit_prereg.md",
        "protocol_sha256": sha256_file(protocol_path),
        "upstream": provenance,
        "config": {
            "seeds": list(SEEDS), "complexities": list(COMPLEXITIES),
            "training_source": "official o6 train only",
            "validation": "per-seed stratified 80/20 development split",
            "max_epochs": args.max_epochs, "patience": args.patience,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "device": str(device),
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "dataset_counts": {
            "train": int(len(train_data["actions"])),
            "tests": {str(c): int(len(d["actions"])) for c, d in tests.items()},
        },
        "methods": methods,
        "summaries": summaries,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "decision": decision["pass"],
        "runtime_seconds": float(time.time() - started),
    }, indent=2))


if __name__ == "__main__":
    main()
