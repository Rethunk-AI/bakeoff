"""Smoke tests for bench.report_cli (bakeoff-report CLI).

Coverage strategy: two critical paths per the issue spec (#36):
  1. --list on an empty store returns zero exit code and no output.
  2. --run-id <id> --format md emits a .md file to --out-dir.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Minimal payload that satisfies emit_reports / rollup requirements.
_FAKE_PAYLOAD = {
    "run_id": "smoke-run",
    "timestamp": "20260604-120000",
    "run_status": "complete",
    "config": {"cost": {"enabled": False}},
    "provenance": {},
    "model_metadata": {},
    "tasks": [{"id": "t0", "domain": "qa", "user_prompt": "hi", "tier": "main"}],
    "records": [
        {
            "model_id": "m_a",
            "task_id": "t0",
            "prompt_id": "plain",
            "tier": "main",
            "quality_heuristic": 1.0,
            "failure_code": None,
            "error": None,
            "wall_clock_seconds": 1.0,
            "seconds_to_first_token": 0.1,
            "tokens_per_second": 10.0,
            "energy_wh": None,
        }
    ],
    "model_scores": [],
    "judgements": [],
    "resumed_from": None,
    "hardware": {},
}


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point BAKEOFF_DATA_DIR at a per-test temp directory."""
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _run_cli(*argv: str) -> int:
    from bench.report_cli import main

    with patch("sys.argv", ["bakeoff-report", *argv]):
        return main()


class TestListCommand:
    def test_list_empty_store_exits_zero(self, capsys):
        rc = _run_cli("--list")
        assert rc == 0
        out, _ = capsys.readouterr()
        assert out == ""

    def test_list_shows_run_ids_newest_first(self, isolated_data_dir):
        from bench import store

        store.write_record("runs", "run-aaa", {"run_id": "run-aaa", "x": 1})
        store.write_record("runs", "run-zzz", {"run_id": "run-zzz", "x": 2})

        rc = _run_cli("--list")
        assert rc == 0
        # Ordering validated by test_list_order_newest_first.

    def test_list_order_newest_first(self, capsys, isolated_data_dir):
        from bench import store

        store.write_record("runs", "run-aaa", {"run_id": "run-aaa"})
        store.write_record("runs", "run-zzz", {"run_id": "run-zzz"})

        rc = _run_cli("--list")
        assert rc == 0
        lines = capsys.readouterr().out.strip().splitlines()
        # list_records returns sorted ascending by stem; reversed = zzz before aaa
        assert lines == ["run-zzz", "run-aaa"]


class TestReportGeneration:
    def test_md_format_emits_md_file(self, tmp_path, isolated_data_dir):
        """--run-id <id> --format md must emit run-<ts>.md to --out-dir."""
        from bench import store

        store.write_record("runs", "smoke-run", _FAKE_PAYLOAD)
        out_dir = tmp_path / "reports"

        rc = _run_cli("--run-id", "smoke-run", "--format", "md", "--out-dir", str(out_dir))
        assert rc == 0

        md_file = out_dir / "run-20260604-120000.md"
        assert md_file.exists(), f"expected {md_file} to be created"
        content = md_file.read_text()
        assert "smoke-run" in content

    def test_html_format_emits_html_file(self, tmp_path, isolated_data_dir):
        from bench import store

        store.write_record("runs", "smoke-run", _FAKE_PAYLOAD)
        out_dir = tmp_path / "reports"

        rc = _run_cli("--run-id", "smoke-run", "--format", "html", "--out-dir", str(out_dir))
        assert rc == 0
        assert (out_dir / "run-20260604-120000.html").exists()

    def test_both_format_emits_md_and_html(self, tmp_path, isolated_data_dir):
        from bench import store

        store.write_record("runs", "smoke-run", _FAKE_PAYLOAD)
        out_dir = tmp_path / "reports"

        rc = _run_cli("--run-id", "smoke-run", "--format", "both", "--out-dir", str(out_dir))
        assert rc == 0
        assert (out_dir / "run-20260604-120000.md").exists()
        assert (out_dir / "run-20260604-120000.html").exists()

    def test_default_run_is_most_recent(self, tmp_path, isolated_data_dir):
        """No --run-id flag: CLI must pick the most recent stored run."""
        from bench import store

        payload_a = {**_FAKE_PAYLOAD, "run_id": "run-aaa", "timestamp": "20260604-110000"}
        payload_z = {**_FAKE_PAYLOAD, "run_id": "run-zzz", "timestamp": "20260604-120000"}
        store.write_record("runs", "run-aaa", payload_a)
        store.write_record("runs", "run-zzz", payload_z)
        out_dir = tmp_path / "reports"

        rc = _run_cli("--format", "md", "--out-dir", str(out_dir))
        assert rc == 0
        # Most recent by ascending stem = run-zzz (last in sorted order)
        assert (out_dir / "run-20260604-120000.md").exists()

    def test_missing_run_id_exits_nonzero(self, tmp_path):
        rc = _run_cli("--run-id", "does-not-exist", "--format", "md", "--out-dir", str(tmp_path))
        assert rc != 0

    def test_empty_store_no_run_id_exits_nonzero(self):
        rc = _run_cli("--format", "md")
        assert rc != 0
