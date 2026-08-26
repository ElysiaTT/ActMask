#!/usr/bin/env python3
"""Run the existing bounded ManiSkill state/GPU smoke in a child process."""
from __future__ import annotations

import json
from pathlib import Path

from actmask.experiments.milestone3p_maniskill_smoke import run


OUTPUT = Path("/data/project/tzh/papers/ActMask/audit/environment/smoke")


if __name__ == "__main__":
    print(json.dumps(run(OUTPUT), indent=2, sort_keys=True))
