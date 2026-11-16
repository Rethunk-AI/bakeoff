"""Regression: schema_version gate and list_runs mtime order."""

from __future__ import annotations

import json
import os

import pytest

from bench.store import SCHEMA_VERSION, StoreError, list_runs, read_record, write_record


def test_read_record_rejects_missing_schema_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "no-version.json").write_text(json.dumps({"id": "no-version"}) + "\n")
    with pytest.raises(StoreError):
        read_record("runs", "no-version")


def test_read_record_rejects_unexpected_schema_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "v99.json").write_text(
        json.dumps({"schema_version": 99, "id": "v99"}) + "\n"
    )
    with pytest.raises(StoreError):
        read_record("runs", "v99")


def test_write_record_stamps_schema_version_so_read_succeeds(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    write_record("runs", "stamped", {"id": "stamped"})
    rec = read_record("runs", "stamped")
    assert rec["schema_version"] == SCHEMA_VERSION


def test_list_runs_orders_by_mtime_not_uuid_stem(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    older_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    newer_id = "00000000-0000-0000-0000-000000000000"
    write_record("runs", older_id, {"id": older_id})
    write_record("runs", newer_id, {"id": newer_id})
    runs = tmp_path / "runs"
    os.utime(runs / f"{older_id}.json", (1_000_000, 1_000_000))
    os.utime(runs / f"{newer_id}.json", (2_000_000, 2_000_000))
    ids = list_runs()
    assert ids[0] == older_id
    assert ids[-1] == newer_id
