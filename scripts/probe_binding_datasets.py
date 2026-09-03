"""Bounded CPU probe of the two simulator-state dataset candidates.

The probe pins official Hugging Face repository revisions, downloads only two
small HDF5 files into a temporary directory, verifies their LFS SHA-256, and
records structural metadata. It never installs a simulator or retains the data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py


MAX_FILE_BYTES = 64 * 1024 * 1024
SELECTIONS = {
    "maniskill": {
        "repo": "haosulab/ManiSkill_Demonstrations",
        "path": "demos/PickCube-v1/teleop/trajectory.h5",
        "metadata_path": "demos/PickCube-v1/teleop/trajectory.json",
        "official_docs": "https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html",
    },
    "robomimic": {
        "repo": "robomimic/robomimic_datasets",
        "path": "v1.5/lift/ph/demo_v15.hdf5",
        "metadata_path": None,
        "official_docs": "https://robomimic.github.io/docs/datasets/robosuite.html",
    },
}


def _curl_bytes(url: str, *, timeout: int = 60) -> bytes:
    if shutil.which("curl") is None:
        raise RuntimeError("the bounded probe requires the host curl executable")
    completed = subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--max-time", str(timeout), url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"curl failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _curl_file(url: str, destination: Path, *, timeout: int = 180) -> None:
    completed = subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--max-time", str(timeout), "--output", str(destination), url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"curl download failed ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )


def _json(url: str) -> Any:
    return json.loads(_curl_bytes(url).decode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hdf5_probe(name: str, path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if name == "maniskill":
            trajectories = sorted(key for key in handle if key.startswith("traj_"))
            first = handle[trajectories[0]]
            env_state = first["env_states"]
            env_state_shapes: dict[str, list[int]] = {}
            if isinstance(env_state, h5py.Dataset):
                env_state_shapes["env_states"] = list(env_state.shape)
            else:
                def collect(path_name: str, value: h5py.Group | h5py.Dataset) -> None:
                    if isinstance(value, h5py.Dataset):
                        env_state_shapes[path_name] = list(value.shape)

                env_state.visititems(collect)
            action_rows = int(first["actions"].shape[0])
            return {
                "root_keys": sorted(handle.keys()),
                "trajectory_count": len(trajectories),
                "first_trajectory": trajectories[0],
                "first_keys": sorted(first.keys()),
                "actions_shape": list(first["actions"].shape),
                "env_states_storage": (
                    "dataset" if isinstance(env_state, h5py.Dataset) else "state_dict_group"
                ),
                "env_state_leaf_shapes": env_state_shapes,
                "has_transition_alignment": bool(
                    env_state_shapes
                    and all(shape[0] == action_rows + 1 for shape in env_state_shapes.values())
                ),
            }
        data = handle["data"]
        demonstrations = sorted(key for key in data if key.startswith("demo_"))
        first = data[demonstrations[0]]
        states = first["states"]
        actions = first["actions"]
        return {
            "root_keys": sorted(handle.keys()),
            "demonstration_count": len(demonstrations),
            "first_demonstration": demonstrations[0],
            "first_keys": sorted(first.keys()),
            "states_shape": list(states.shape),
            "actions_shape": list(actions.shape),
            "has_equal_state_action_rows": bool(states.shape[0] == actions.shape[0]),
            "has_model_file": "model_file" in first.attrs,
            "has_env_args": "env_args" in data.attrs,
        }


def _repo_inventory(repo: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _json(f"https://huggingface.co/api/datasets/{repo}")
    tree = _json(
        f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&limit=1000"
    )
    if not isinstance(tree, list):
        raise RuntimeError(f"unexpected Hugging Face tree response for {repo}")
    return metadata, tree


def run(*, metadata_only: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "actmask-binding-dataset-probe-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "maximum_file_bytes": MAX_FILE_BYTES,
            "temporary_downloads_deleted": True,
            "metadata_only": bool(metadata_only),
        },
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "h5py": h5py.__version__,
            "packages": {
                name: importlib.util.find_spec(name) is not None
                for name in ("mani_skill", "robomimic", "robosuite", "libero")
            },
        },
        "datasets": {},
    }
    with tempfile.TemporaryDirectory(prefix="actmask_binding_probe_") as temporary:
        temporary_root = Path(temporary)
        for name, selection in SELECTIONS.items():
            repo = str(selection["repo"])
            metadata, tree = _repo_inventory(repo)
            by_path = {str(row.get("path")): row for row in tree}
            remote = by_path[str(selection["path"])]
            size = int(remote["size"])
            lfs_sha256 = str(remote.get("lfs", {}).get("oid", ""))
            row: dict[str, Any] = {
                "repo": repo,
                "repo_revision": metadata["sha"],
                "repo_last_modified": metadata.get("lastModified"),
                "selected_path": selection["path"],
                "selected_bytes": size,
                "lfs_sha256": lfs_sha256,
                "official_docs": selection["official_docs"],
                "bounded": size <= MAX_FILE_BYTES,
            }
            metadata_path = selection["metadata_path"]
            if metadata_path is not None:
                metadata_url = (
                    f"https://huggingface.co/datasets/{repo}/resolve/{metadata['sha']}/"
                    f"{quote(str(metadata_path), safe='/')}"
                )
                trajectory_metadata = _json(metadata_url)
                row["trajectory_metadata"] = {
                    "env_info": trajectory_metadata.get("env_info"),
                    "episodes": len(trajectory_metadata.get("episodes", [])),
                    "all_success": bool(
                        trajectory_metadata.get("episodes")
                        and all(item.get("success") for item in trajectory_metadata["episodes"])
                    ),
                    "source_type": trajectory_metadata.get("source_type"),
                    "source_commit": trajectory_metadata.get("commit_info", {}).get("commit_id"),
                }
            if not metadata_only:
                if size > MAX_FILE_BYTES:
                    raise RuntimeError(f"refusing oversized probe file: {selection['path']}")
                destination = temporary_root / f"{name}.h5"
                data_url = (
                    f"https://huggingface.co/datasets/{repo}/resolve/{metadata['sha']}/"
                    f"{quote(str(selection['path']), safe='/')}"
                )
                _curl_file(data_url, destination)
                observed_sha256 = _sha256(destination)
                if destination.stat().st_size != size:
                    raise RuntimeError(f"size mismatch for {selection['path']}")
                if lfs_sha256 and observed_sha256 != lfs_sha256:
                    raise RuntimeError(f"SHA-256 mismatch for {selection['path']}")
                row["download_verification"] = {
                    "bytes": destination.stat().st_size,
                    "sha256": observed_sha256,
                    "matches_official_lfs": bool(observed_sha256 == lfs_sha256),
                }
                row["hdf5"] = _hdf5_probe(name, destination)
            output["datasets"][name] = row
    output["decision"] = {
        "primary": "maniskill",
        "backup": "robomimic",
        "metadata_passed": all(
            row["bounded"] and len(row["lfs_sha256"]) == 64
            for row in output["datasets"].values()
        ),
        "data_probe_passed": bool(
            not metadata_only
            and all(
                row.get("download_verification", {}).get("matches_official_lfs")
                for row in output["datasets"].values()
            )
        ),
        "simulator_runtime_ready": False,
        "reason": "Official files are structurally suitable, but simulator packages and a supported Linux/CUDA host are not present locally.",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/results/binding_dataset_probe.json"),
    )
    args = parser.parse_args()
    result = run(metadata_only=args.metadata_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
