"""Status ENUM types in schema.sql stay in step with the values the code writes."""

from __future__ import annotations

import re
from pathlib import Path

SCHEMA = (Path(__file__).resolve().parent.parent / "schema" / "schema.sql").read_text()


def _enum(name: str) -> set[str]:
    m = re.search(rf"CREATE TYPE {name}\s+AS ENUM \((.*?)\);", SCHEMA)
    assert m, f"no enum {name}"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_run_status_matches_published_values():
    assert _enum("run_status") == {"complete", "incomplete", "failed"}


def test_status_columns_use_the_enums_not_text_checks():
    assert not re.search(r"status\s+TEXT", SCHEMA)
    assert "CHECK (status IN" not in SCHEMA
