"""Versioned, provenance-checked fair-baseline caching for Milestone 2E-Final.

The original Milestone 2E runner kept a small, candidate-count-only cache.
That is insufficient for the final protocol: an analytic baseline can change
when the observable view, corruption, candidate-world membership, metric
schema, or code changes even when the candidate count does not.  This module
is deliberately independent of the legacy output tree and provides a
reusable cache contract for all final-v2 evaluation consumers.

The cache is fail-closed.  A file is usable only when its complete canonical
key and embedded digest match the requested key.  A changed configuration or
metric schema therefore yields a cache miss, while a file at the expected
path with inconsistent metadata raises :class:`StaleFairBaselineCacheError`.
Historical NDCG caches cannot be silently reused for corrected-NDCG reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch


CACHE_SCHEMA_VERSION = "milestone2e-final-fair-baseline-cache-v1"
"""On-disk envelope schema.  Bump it for an incompatible cache format."""

# This is deliberately distinct from the historical, non-standard NDCG
# artifacts.  It matches ``actmask.experiments.milestone2e.RANKING_METRIC_SCHEMA``
# without importing the larger evaluation module into this cache primitive.
CORRECTED_NDCG_METRIC_SCHEMA_VERSION = "milestone2e-ranking-v2-standard-ndcg-utility-gain"
LEGACY_NDCG_METRIC_SCHEMA_VERSION = "milestone2e-ranking-v1-legacy-oracle-discount"
FINAL_EVALUATION_SCHEMA_VERSION = "milestone2e-final-evaluation-v2"

PayloadT = TypeVar("PayloadT")


class FairBaselineCacheError(RuntimeError):
    """Base class for final fair-baseline cache failures."""


class StaleFairBaselineCacheError(FairBaselineCacheError):
    """Raised when an on-disk entry cannot prove it matches the requested key."""


class FairBaselineCacheLockTimeout(FairBaselineCacheError):
    """Raised instead of duplicating a long baseline computation after a lock timeout."""


def _json_value(value: Any) -> Any:
    """Convert supported provenance values to deterministic JSON-native values.

    Configuration and provenance must be hashed without relying on Python's
    mapping order, tensor representation, or platform-specific path objects.
    Unsupported values intentionally fail rather than silently falling back to
    ``str(value)`` (which could conceal a materially different configuration).
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("cache provenance cannot contain non-finite floats")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("cache provenance tensors must be moved to CPU first")
        return _json_value(value.detach().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported cache provenance value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return stable JSON for cache keys, config fingerprints, and validation."""

    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 digest of a canonical cache-provenance value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty_identifier(value: Any, *, field: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


@dataclass(frozen=True)
class FairBaselineWorld:
    """The ordered candidate set belonging to one base observable world.

    Candidate IDs are deliberately retained in their evaluation order.  The
    fair baseline payload is aligned to that order, so sorting IDs would turn
    a candidate-order change into an undetected cache hit.
    """

    world_id: str
    candidate_set_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_id", _nonempty_identifier(self.world_id, field="world_id"))
        object.__setattr__(self, "candidate_set_id", _nonempty_identifier(self.candidate_set_id, field="candidate_set_id"))
        identifiers = tuple(_nonempty_identifier(value, field="candidate_id") for value in self.candidate_ids)
        if not identifiers:
            raise ValueError("a baseline world requires at least one candidate")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"candidate IDs are duplicated in world {self.world_id!r}")
        object.__setattr__(self, "candidate_ids", identifiers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "candidate_set_id": self.candidate_set_id,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class FairBaselineCacheKey:
    """Complete provenance key for one deterministic fair-baseline payload.

    ``config_hash`` should be derived from the complete frozen evaluation
    configuration.  ``code_provenance`` must include a source digest or a
    version-control revision when source is available.  Both are part of the
    digest, so changed code/configuration cannot reuse a prior payload.
    """

    baseline_name: str
    baseline_parameters: Mapping[str, Any]
    config_hash: str
    code_provenance: Mapping[str, Any]
    evaluation_schema_version: str
    metric_schema_version: str
    split_id: str
    observation_view_id: str
    corruption_id: str
    candidate_count: int
    worlds: tuple[FairBaselineWorld, ...]
    schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_name", _nonempty_identifier(self.baseline_name, field="baseline_name"))
        normalized_parameters = _json_value(dict(self.baseline_parameters))
        if not isinstance(normalized_parameters, dict):
            raise ValueError("baseline_parameters must be a JSON mapping")
        object.__setattr__(self, "baseline_parameters", json.loads(canonical_json(normalized_parameters)))
        object.__setattr__(self, "config_hash", _nonempty_identifier(self.config_hash, field="config_hash"))
        object.__setattr__(
            self,
            "evaluation_schema_version",
            _nonempty_identifier(self.evaluation_schema_version, field="evaluation_schema_version"),
        )
        object.__setattr__(
            self,
            "metric_schema_version",
            _nonempty_identifier(self.metric_schema_version, field="metric_schema_version"),
        )
        object.__setattr__(self, "split_id", _nonempty_identifier(self.split_id, field="split_id"))
        object.__setattr__(
            self,
            "observation_view_id",
            _nonempty_identifier(self.observation_view_id, field="observation_view_id"),
        )
        object.__setattr__(self, "corruption_id", _nonempty_identifier(self.corruption_id, field="corruption_id"))
        object.__setattr__(self, "schema_version", _nonempty_identifier(self.schema_version, field="schema_version"))
        if int(self.candidate_count) <= 0:
            raise ValueError("candidate_count must be positive")
        object.__setattr__(self, "candidate_count", int(self.candidate_count))
        worlds = tuple(self.worlds)
        if not worlds:
            raise ValueError("a fair-baseline cache key requires at least one base world")
        if any(not isinstance(world, FairBaselineWorld) for world in worlds):
            raise TypeError("worlds must contain FairBaselineWorld entries")
        if len({world.world_id for world in worlds}) != len(worlds):
            raise ValueError("world IDs must be unique within one cache entry")
        if len({world.candidate_set_id for world in worlds}) != len(worlds):
            raise ValueError("candidate-set IDs must be unique within one cache entry")
        mismatched = [world.world_id for world in worlds if len(world.candidate_ids) != self.candidate_count]
        if mismatched:
            raise ValueError(
                "candidate_count does not match every candidate world: " + ", ".join(mismatched[:3])
            )
        object.__setattr__(self, "worlds", worlds)
        normalized_provenance = _json_value(dict(self.code_provenance))
        if not isinstance(normalized_provenance, dict) or not normalized_provenance:
            raise ValueError("code_provenance must be a non-empty JSON mapping")
        if not any(
            key in normalized_provenance
            for key in ("git_revision", "source_tree_sha256", "source_files_sha256", "code_version")
        ):
            raise ValueError(
                "code_provenance must include git_revision, source_tree_sha256, source_files_sha256, or code_version"
            )
        # Copy through JSON so later mutation of the caller's mapping cannot
        # change a frozen key object's semantics.
        object.__setattr__(self, "code_provenance", json.loads(canonical_json(normalized_provenance)))

    @classmethod
    def from_context(
        cls,
        *,
        baseline_name: str,
        config: Mapping[str, Any],
        code_provenance: Mapping[str, Any],
        worlds: Sequence[FairBaselineWorld],
        split_id: str,
        observation_view_id: str,
        corruption_id: str,
        candidate_count: int,
        baseline_parameters: Mapping[str, Any] | None = None,
        evaluation_schema_version: str = FINAL_EVALUATION_SCHEMA_VERSION,
        metric_schema_version: str = CORRECTED_NDCG_METRIC_SCHEMA_VERSION,
        schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> "FairBaselineCacheKey":
        """Build a key from a complete config mapping and frozen world provenance."""

        return cls(
            baseline_name=baseline_name,
            baseline_parameters={} if baseline_parameters is None else baseline_parameters,
            config_hash=canonical_hash(config),
            code_provenance=code_provenance,
            evaluation_schema_version=evaluation_schema_version,
            metric_schema_version=metric_schema_version,
            split_id=split_id,
            observation_view_id=observation_view_id,
            corruption_id=corruption_id,
            candidate_count=candidate_count,
            worlds=tuple(worlds),
            schema_version=schema_version,
        )

    @property
    def world_ids(self) -> tuple[str, ...]:
        """Ordered base-world IDs included in this deterministic payload."""

        return tuple(world.world_id for world in self.worlds)

    @property
    def candidate_set_ids(self) -> tuple[str, ...]:
        """Ordered candidate-set IDs aligned with :attr:`world_ids`."""

        return tuple(world.candidate_set_id for world in self.worlds)

    @property
    def candidate_ids_by_world(self) -> tuple[tuple[str, ...], ...]:
        """Candidate IDs in evaluation order for every base world."""

        return tuple(world.candidate_ids for world in self.worlds)

    def as_dict(self) -> dict[str, Any]:
        """Return the complete canonical material used to determine a cache hit."""

        return {
            "schema_version": self.schema_version,
            "baseline_name": self.baseline_name,
            "baseline_parameters": _json_value(self.baseline_parameters),
            "config_hash": self.config_hash,
            "code_provenance": _json_value(self.code_provenance),
            "evaluation_schema_version": self.evaluation_schema_version,
            "metric_schema_version": self.metric_schema_version,
            "split_id": self.split_id,
            "observation_view_id": self.observation_view_id,
            "corruption_id": self.corruption_id,
            "candidate_count": self.candidate_count,
            "world_ids": list(self.world_ids),
            "candidate_set_ids": list(self.candidate_set_ids),
            "candidate_ids_by_world": [list(ids) for ids in self.candidate_ids_by_world],
            "worlds": [world.as_dict() for world in self.worlds],
        }

    @property
    def digest(self) -> str:
        """Stable identity used in the on-disk filename and envelope validation."""

        return canonical_hash(self.as_dict())


def worlds_from_metadata(
    metadata: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    world_id_field: str = "group_id",
    candidate_set_id_field: str = "candidate_set_id",
    candidate_id_field: str = "candidate_id",
) -> tuple[FairBaselineWorld, ...]:
    """Derive ordered candidate-world provenance from frozen evaluation rows.

    The function fails if a row is missing any identity field, if a world does
    not contain exactly ``candidate_count`` candidates, or if one candidate ID
    occurs twice.  It is therefore safe to use immediately before calculating
    a baseline over the same dataset/order.
    """

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive")
    groups: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
    for row in metadata:
        missing = [
            field
            for field in (world_id_field, candidate_set_id_field, candidate_id_field)
            if field not in row or row[field] is None
        ]
        if missing:
            raise ValueError("metadata is missing cache identity field(s): " + ", ".join(missing))
        world_id = _nonempty_identifier(row[world_id_field], field=world_id_field)
        candidate_set_id = _nonempty_identifier(row[candidate_set_id_field], field=candidate_set_id_field)
        candidate_id = _nonempty_identifier(row[candidate_id_field], field=candidate_id_field)
        groups.setdefault((world_id, candidate_set_id), []).append(candidate_id)
    worlds = tuple(
        FairBaselineWorld(world_id=world_id, candidate_set_id=candidate_set_id, candidate_ids=tuple(candidate_ids))
        for (world_id, candidate_set_id), candidate_ids in groups.items()
    )
    if not worlds:
        raise ValueError("metadata did not contain any candidate worlds")
    mismatched = [world.world_id for world in worlds if len(world.candidate_ids) != int(candidate_count)]
    if mismatched:
        raise ValueError(
            "metadata candidate count differs from requested candidate_count for world(s): "
            + ", ".join(mismatched[:3])
        )
    return worlds


def code_provenance_for_files(
    source_files: Sequence[str | Path], *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Collect source-file hashes and an optional Git revision for a cache key.

    Git is advisory: a source digest is always included, while an unavailable
    repository or Git executable simply yields ``git_revision: None``.  Paths
    are represented relative to ``repository_root`` when possible so the same
    project copied to another absolute location retains the same provenance.
    """

    if not source_files:
        raise ValueError("at least one source file is required for code provenance")
    root = Path(repository_root).resolve() if repository_root is not None else None
    digests: dict[str, str] = {}
    for raw_path in source_files:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source provenance file does not exist: {path}")
        try:
            name = str(path.relative_to(root)) if root is not None else str(path)
        except ValueError:
            name = str(path)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    git_revision: str | None = None
    if root is not None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            revision = result.stdout.strip()
            git_revision = revision or None
    normalized = {name: digests[name] for name in sorted(digests)}
    return {
        "git_revision": git_revision,
        "source_files_sha256": normalized,
        "source_tree_sha256": canonical_hash(normalized),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
    }


@dataclass(frozen=True)
class FairBaselineCacheResult(Generic[PayloadT]):
    """Payload plus enough evidence to record whether it was recomputed."""

    payload: PayloadT
    cache_hit: bool
    path: Path
    key_digest: str
    created_at_utc: str


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "baseline"


class FairBaselineCache:
    """Atomic, provenance-validated cache for deterministic fair baselines.

    ``root`` should be inside the new versioned output tree, e.g.
    ``outputs/actmask/milestone2e_final_v2/fair_baseline_cache``.  This class
    never reads or writes the legacy ``milestone2e`` cache directory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        lock_timeout_seconds: float = 6.0 * 60.0 * 60.0,
        lock_poll_seconds: float = 0.05,
    ) -> None:
        self.root = Path(root)
        if float(lock_timeout_seconds) <= 0.0:
            raise ValueError("lock_timeout_seconds must be positive")
        if float(lock_poll_seconds) <= 0.0:
            raise ValueError("lock_poll_seconds must be positive")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.lock_poll_seconds = float(lock_poll_seconds)

    def path_for(self, key: FairBaselineCacheKey) -> Path:
        """Return the deterministic final-v2 location for ``key`` without I/O."""

        return (
            self.root
            / _safe_filename(key.schema_version)
            / _safe_filename(key.baseline_name)
            / f"{key.digest}.pt"
        )

    def _lock_path_for(self, key: FairBaselineCacheKey) -> Path:
        return self.path_for(key).with_suffix(".lock")

    @staticmethod
    def _validate_timestamp(value: Any, *, path: Path) -> str:
        if not isinstance(value, str) or not value:
            raise StaleFairBaselineCacheError(f"cache entry lacks creation timestamp: {path}")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise StaleFairBaselineCacheError(f"cache entry has invalid creation timestamp: {path}") from error
        return value

    def _read(self, key: FairBaselineCacheKey, path: Path) -> FairBaselineCacheResult[Any]:
        try:
            envelope = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # corrupt/partial serializations are not safe cache misses
            raise StaleFairBaselineCacheError(f"cannot safely load fair-baseline cache entry: {path}") from error
        if not isinstance(envelope, Mapping):
            raise StaleFairBaselineCacheError(f"fair-baseline cache entry is not a mapping: {path}")
        required = {
            "schema_version",
            "cache_key",
            "cache_key_digest",
            "created_at_utc",
            "baseline_name",
            "baseline_parameters",
            "config_hash",
            "code_provenance",
            "evaluation_schema_version",
            "metric_schema_version",
            "split_id",
            "world_ids",
            "candidate_set_ids",
            "candidate_ids_by_world",
            "candidate_count",
            "observation_view_id",
            "corruption_id",
            "payload",
        }
        missing = sorted(required.difference(envelope))
        if missing:
            raise StaleFairBaselineCacheError(
                f"fair-baseline cache entry is missing required provenance {missing}: {path}"
            )
        expected = key.as_dict()
        recorded_key = envelope["cache_key"]
        if not isinstance(recorded_key, Mapping) or canonical_json(recorded_key) != canonical_json(expected):
            raise StaleFairBaselineCacheError(f"fair-baseline cache key mismatch: {path}")
        if envelope["cache_key_digest"] != key.digest:
            raise StaleFairBaselineCacheError(f"fair-baseline cache digest mismatch: {path}")
        # These direct fields make the cache self-describing for audit tools;
        # validate them separately so a hand-edited nested key cannot hide a
        # stale file from a lightweight metadata inspection.
        direct_expected = {
            "schema_version": key.schema_version,
            "baseline_name": key.baseline_name,
            "baseline_parameters": _json_value(key.baseline_parameters),
            "config_hash": key.config_hash,
            "code_provenance": _json_value(key.code_provenance),
            "evaluation_schema_version": key.evaluation_schema_version,
            "metric_schema_version": key.metric_schema_version,
            "split_id": key.split_id,
            "world_ids": list(key.world_ids),
            "candidate_set_ids": list(key.candidate_set_ids),
            "candidate_ids_by_world": [list(ids) for ids in key.candidate_ids_by_world],
            "candidate_count": key.candidate_count,
            "observation_view_id": key.observation_view_id,
            "corruption_id": key.corruption_id,
        }
        mismatch = [
            name for name, expected_value in direct_expected.items()
            if canonical_json(envelope[name]) != canonical_json(expected_value)
        ]
        if mismatch:
            raise StaleFairBaselineCacheError(
                f"fair-baseline cache direct provenance mismatch ({', '.join(mismatch)}): {path}"
            )
        created_at_utc = self._validate_timestamp(envelope["created_at_utc"], path=path)
        return FairBaselineCacheResult(
            payload=envelope["payload"],
            cache_hit=True,
            path=path,
            key_digest=key.digest,
            created_at_utc=created_at_utc,
        )

    def get(self, key: FairBaselineCacheKey) -> FairBaselineCacheResult[Any] | None:
        """Return a verified entry, ``None`` for a true miss, or raise if stale."""

        path = self.path_for(key)
        return self._read(key, path) if path.exists() else None

    def put(self, key: FairBaselineCacheKey, payload: PayloadT) -> FairBaselineCacheResult[PayloadT]:
        """Store a new entry atomically; never overwrite an existing valid entry."""

        existing = self.get(key)
        if existing is not None:
            return FairBaselineCacheResult(
                payload=existing.payload,
                cache_hit=True,
                path=existing.path,
                key_digest=existing.key_digest,
                created_at_utc=existing.created_at_utc,
            )
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        created_at_utc = _utc_timestamp()
        envelope = {
            "schema_version": key.schema_version,
            "cache_key": key.as_dict(),
            "cache_key_digest": key.digest,
            "created_at_utc": created_at_utc,
            "baseline_name": key.baseline_name,
            "baseline_parameters": _json_value(key.baseline_parameters),
            "config_hash": key.config_hash,
            "code_provenance": _json_value(key.code_provenance),
            "evaluation_schema_version": key.evaluation_schema_version,
            "metric_schema_version": key.metric_schema_version,
            "split_id": key.split_id,
            "world_ids": list(key.world_ids),
            "candidate_set_ids": list(key.candidate_set_ids),
            "candidate_ids_by_world": [list(ids) for ids in key.candidate_ids_by_world],
            "candidate_count": key.candidate_count,
            "observation_view_id": key.observation_view_id,
            "corruption_id": key.corruption_id,
            "payload": payload,
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(envelope, temporary)
            try:
                # A hard link is an atomic no-clobber publish on the same
                # filesystem.  Even a caller using ``put`` directly therefore
                # cannot replace another process's completed baseline.
                os.link(temporary, path)
            except FileExistsError:
                existing = self.get(key)
                if existing is None:  # pragma: no cover - defensive filesystem race guard
                    raise FairBaselineCacheError(f"cache entry vanished while storing: {path}")
                return FairBaselineCacheResult(
                    payload=existing.payload,
                    cache_hit=True,
                    path=existing.path,
                    key_digest=existing.key_digest,
                    created_at_utc=existing.created_at_utc,
                )
        finally:
            if temporary.exists():
                temporary.unlink()
        return FairBaselineCacheResult(
            payload=payload,
            cache_hit=False,
            path=path,
            key_digest=key.digest,
            created_at_utc=created_at_utc,
        )

    def _acquire_lock(self, key: FairBaselineCacheKey) -> int:
        """Acquire one per-key lock without duplicating expensive baseline work."""

        path = self._lock_path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise FairBaselineCacheLockTimeout(
                        f"timed out waiting for fair-baseline cache lock: {path}"
                    )
                time.sleep(self.lock_poll_seconds)
                continue
            try:
                os.write(descriptor, f"pid={os.getpid()} created_at_utc={_utc_timestamp()}\n".encode("utf-8"))
            except Exception:
                os.close(descriptor)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise
            return descriptor

    def _release_lock(self, key: FairBaselineCacheKey, descriptor: int) -> None:
        os.close(descriptor)
        try:
            self._lock_path_for(key).unlink()
        except FileNotFoundError:
            pass

    def get_or_compute(
        self, key: FairBaselineCacheKey, compute: Callable[[], PayloadT]
    ) -> FairBaselineCacheResult[PayloadT]:
        """Use a verified hit or calculate/store exactly one deterministic payload.

        The second lookup after acquiring the key lock makes independently
        started final-evaluation processes share a completed fair baseline
        rather than recomputing it for every checkpoint or seed.
        """

        existing = self.get(key)
        if existing is not None:
            return FairBaselineCacheResult(
                payload=existing.payload,
                cache_hit=True,
                path=existing.path,
                key_digest=existing.key_digest,
                created_at_utc=existing.created_at_utc,
            )
        descriptor = self._acquire_lock(key)
        try:
            existing = self.get(key)
            if existing is not None:
                return FairBaselineCacheResult(
                    payload=existing.payload,
                    cache_hit=True,
                    path=existing.path,
                    key_digest=existing.key_digest,
                    created_at_utc=existing.created_at_utc,
                )
            return self.put(key, compute())
        finally:
            self._release_lock(key, descriptor)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CORRECTED_NDCG_METRIC_SCHEMA_VERSION",
    "FINAL_EVALUATION_SCHEMA_VERSION",
    "LEGACY_NDCG_METRIC_SCHEMA_VERSION",
    "FairBaselineCache",
    "FairBaselineCacheError",
    "FairBaselineCacheKey",
    "FairBaselineCacheLockTimeout",
    "FairBaselineCacheResult",
    "FairBaselineWorld",
    "StaleFairBaselineCacheError",
    "canonical_hash",
    "canonical_json",
    "code_provenance_for_files",
    "worlds_from_metadata",
]
