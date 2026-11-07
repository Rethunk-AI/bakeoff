"""Status ENUM types in schema.sql stay in step with the values the code writes."""

from __future__ import annotations

import re
from pathlib import Path

from bench import queue

SCHEMA = (Path(__file__).resolve().parent.parent / "schema" / "schema.sql").read_text()


def _enum(name: str) -> set[str]:
    m = re.search(rf"CREATE TYPE {name}\s+AS ENUM \((.*?)\);", SCHEMA)
    assert m, f"no enum {name}"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_run_queue_status_matches_queue_constants():
    used = {v for k, v in vars(queue).items() if k.startswith("_STATUS_") and isinstance(v, str)}
    assert _enum("run_queue_status") == used


def test_runner_status_covers_the_states_the_queue_writes():
    assert {queue._RUNNER_ACTIVE, queue._RUNNER_DEAD} <= _enum("runner_status")


def test_run_status_matches_published_values():
    assert _enum("run_status") == {"complete", "incomplete", "failed"}


def test_status_columns_use_the_enums_not_text_checks():
    assert not re.search(r"status\s+TEXT", SCHEMA)
    assert "CHECK (status IN" not in SCHEMA
