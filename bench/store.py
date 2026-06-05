"""Disk-persistence primitives for the bakeoff harness.

Directory-per-table, UUID-filename JSON records under BAKEOFF_DATA_DIR.
Writes are atomic (temp file + os.replace). Audit fields (schema_version,
created_at, updated_at) are stamped automatically on every write.

Layout::

    <BAKEOFF_DATA_DIR>/
      models/<uuid>.json
      tasks/<natural_key_hash>.json
      prompts/<content_sha256>.json
      runners/<runner_id>.json
      runs/<run_id>.json
      run_queue/pending/<queue_id>.json
      run_queue/completed/<queue_id>.json

BAKEOFF_DATA_DIR is resolved at call time from the environment so that
monkeypatching in tests works correctly. It defaults to ~/.local/share/bakeoff.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from bench.constants import BAKEOFF_CREATOR_NAMESPACE, BAKEOFF_MODEL_NAMESPACE

SCHEMA_VERSION = 1
_DEFAULT_DATA_DIR = "~/.local/share/bakeoff"


class StoreError(ValueError):
    """Raised when a store operation cannot be completed."""


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------


def data_dir() -> Path:
    """Resolve the bakeoff data directory.

    Reads BAKEOFF_DATA_DIR from the environment at call time (not module
    import time) so that monkeypatching in tests has effect. Expands ~ and
    returns an absolute Path.
    """
    raw = os.environ.get("BAKEOFF_DATA_DIR", _DEFAULT_DATA_DIR)
    return Path(os.path.expanduser(raw)).resolve()


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_audit(existing: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with audit fields set.

    - ``schema_version`` is set to SCHEMA_VERSION if absent.
    - ``created_at`` is preserved from *existing* on re-writes; set on first write.
    - ``updated_at`` is always refreshed.
    """
    out = dict(record)
    now = _utc_now()
    if "schema_version" not in out:
        out["schema_version"] = SCHEMA_VERSION
    if existing and existing.get("created_at"):
        out["created_at"] = existing["created_at"]
    else:
        out.setdefault("created_at", now)
    out["updated_at"] = now
    return out


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------


def _table_dir(table: str) -> Path:
    return data_dir() / table


def write_record(table: str, record_id: str, data: dict[str, Any]) -> Path:
    """Write *data* as JSON to ``<data_dir>/<table>/<record_id>.json`` atomically.

    - Creates the table directory on demand.
    - Stamps audit fields (schema_version, created_at, updated_at).
    - Uses a temp file + os.replace for atomic writes so concurrent readers
      never see a partial file.

    Returns the path to the written file.
    """
    table_path = _table_dir(table)
    table_path.mkdir(parents=True, exist_ok=True)
    target = table_path / f"{record_id}.json"

    # Preserve created_at from an existing record.
    existing: dict[str, Any] | None = None
    if target.exists():
        try:
            with target.open() as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = None

    stamped = _stamp_audit(existing, data)

    # Write to a temp file in the same directory, then atomically replace.
    fd, tmp_path = tempfile.mkstemp(dir=table_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(stamped, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    return target


def read_record(table: str, record_id: str) -> dict[str, Any]:
    """Read and return the JSON record at ``<data_dir>/<table>/<record_id>.json``.

    Raises StoreError if the file is missing or not valid JSON.
    """
    target = _table_dir(table) / f"{record_id}.json"
    try:
        with target.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        raise StoreError(f"record not found: {table}/{record_id}") from None
    except json.JSONDecodeError as e:
        raise StoreError(f"invalid JSON in {table}/{record_id}: {e}") from e
    if not isinstance(data, dict):
        raise StoreError(f"{table}/{record_id} must contain a JSON object")
    return data


def list_records(table: str) -> list[str]:
    """Return a sorted list of record IDs (stem of .json files) in *table*.

    Returns an empty list if the table directory does not exist.
    """
    table_path = _table_dir(table)
    if not table_path.is_dir():
        return []
    return sorted(p.stem for p in table_path.glob("*.json"))


def list_runs() -> list[str]:
    """Return a sorted list of stored run IDs (newest-first by stem).

    Thin wrapper over ``list_records("runs")`` so callers don't repeat
    the table name string.
    """
    return list_records("runs")


def delete_record(table: str, record_id: str) -> None:
    """Delete the record at ``<data_dir>/<table>/<record_id>.json``.

    Raises StoreError if the file does not exist.
    """
    target = _table_dir(table) / f"{record_id}.json"
    try:
        target.unlink()
    except FileNotFoundError:
        raise StoreError(f"record not found: {table}/{record_id}") from None


# ---------------------------------------------------------------------------
# UUID5 helpers (importing from bench.constants)
# ---------------------------------------------------------------------------


def model_uuid(model_hash: str) -> str:
    """Deterministic UUID for a model with a known weights hash.

    UUID5(BAKEOFF_MODEL_NAMESPACE, model_hash).
    """
    return str(uuid5(BAKEOFF_MODEL_NAMESPACE, model_hash))


def provisional_model_uuid(
    source_url: str,
    parameter_count_b: float | None,
    source_size: int | None,
) -> str:
    """Provisional UUID for a model before the weights hash is known.

    UUID5(BAKEOFF_MODEL_NAMESPACE, "{url}|{params}|{size}").
    Matches the dedup key pattern from constants.py and schema.sql.
    """
    key = f"{source_url}|{parameter_count_b}|{source_size}"
    return str(uuid5(BAKEOFF_MODEL_NAMESPACE, key))


def creator_uuid(homepage: str) -> str:
    """Deterministic UUID for a creator with a known homepage.

    UUID5(BAKEOFF_CREATOR_NAMESPACE, homepage).
    """
    return str(uuid5(BAKEOFF_CREATOR_NAMESPACE, homepage))


def provisional_creator_uuid(display_name: str) -> str:
    """Provisional UUID for a creator before the homepage is confirmed.

    UUID5(BAKEOFF_CREATOR_NAMESPACE, display_name).
    """
    return str(uuid5(BAKEOFF_CREATOR_NAMESPACE, display_name))


# Expose namespaces for external use (e.g. tests).
MODEL_NAMESPACE: UUID = BAKEOFF_MODEL_NAMESPACE
CREATOR_NAMESPACE: UUID = BAKEOFF_CREATOR_NAMESPACE
