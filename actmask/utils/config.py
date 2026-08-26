"""Configuration helpers with project-root-relative path handling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to the ActMask project root.

    This deliberately does not depend on the process working directory, so the
    module CLIs behave the same when launched from the repository or elsewhere.
    """

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path, accepting either cwd- or project-relative input."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return resolve_project_path(candidate)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"ActMask config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"ActMask config must be a mapping, got {type(loaded).__name__}")
    return deepcopy(dict(loaded))
