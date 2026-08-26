"""Run the Milestone 3P state-only ManiSkill GPU pilot."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from actmask.data.maniskill_pilot import generate_pilot


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "actmask" / "milestone3p_gpu_benchmark" / "maniskill_pilot"
    summary = generate_pilot(output_dir)
    runtime = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    (output_dir / "runtime_manifest.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
