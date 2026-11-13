"""Smoke tests for bench.runner dry-run path.

These tests exercise the full dry-run code path (config load → validation →
dataset generation → proxy config generation) without starting a proxy or
making any network calls. They serve as CI fixtures that catch regressions
in non-default config branches (alternate judge modes, judge disabled, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Minimal valid config shared by multiple tests.
_BASE_CFG = {
    "run": {"seed": 42},
    "engined": {
        "spec_dir": "/usr/local/src/com.github/Rethunk-Tech/engined/engines/llama",
        "models_dir": "/tmp/models",
    },
    "server": {
        "ctx": 4096,
        "ngl": 99,
    },
    "dataset": {"n": 5, "domains": ["qa"]},
    "models": [
        {"id": "m_a", "gguf": "org/repo-GGUF/model-a-Q4_K_M.gguf"},
        {"id": "m_b", "gguf": "org/repo-GGUF/model-b-Q4_K_M.gguf"},
    ],
    "prompts": [{"id": "plain", "system": "Be helpful."}],
    "judge": {
        "enabled": True,
        "mode": "pairwise_all",
        "gguf": "org/repo-GGUF/judge-Q4_K_M.gguf",
        "ctx": 8192,
    },
    "cost": {"enabled": False},
    "output": {"dir": "results", "emit_markdown": False, "emit_html": False},
}


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run_dry(config_path: Path, capsys) -> int:
    from bench.runner import main

    with (
        patch("sys.argv", ["runner", "--config", str(config_path), "--dry-run"]),
        patch("bench.runner.write_jsonl"),
    ):
        return main()


class TestDryRunSmoke:
    def test_default_config_exits_zero(self, tmp_path, capsys):
        p = _write_config(tmp_path, _BASE_CFG)
        rc = _run_dry(p, capsys)
        assert rc == 0

    def test_dry_run_summary_in_stderr(self, tmp_path, capsys):
        p = _write_config(tmp_path, _BASE_CFG)
        _run_dry(p, capsys)
        err = capsys.readouterr().err
        assert "[dry-run] ok" in err
        assert "validation passed" in err
        assert "tasks" in err
        assert "prompts" in err

    def test_judge_mode_in_summary(self, tmp_path, capsys):
        p = _write_config(tmp_path, _BASE_CFG)
        _run_dry(p, capsys)
        err = capsys.readouterr().err
        assert "pairwise_all" in err

    def test_judge_disabled_shows_off(self, tmp_path, capsys):
        cfg = {**_BASE_CFG, "judge": {"enabled": False}}
        p = _write_config(tmp_path, cfg)
        _run_dry(p, capsys)
        err = capsys.readouterr().err
        assert "judge=off" in err

    def test_scored_judge_in_summary(self, tmp_path, capsys):
        cfg = {
            **_BASE_CFG,
            "judge": {
                "enabled": True,
                "mode": "scored",
                "gguf": "org/repo-GGUF/judge-Q4_K_M.gguf",
                "ctx": 4096,
            },
        }
        p = _write_config(tmp_path, cfg)
        _run_dry(p, capsys)
        err = capsys.readouterr().err
        assert "scored" in err

    def test_invalid_config_exits_nonzero(self, tmp_path, capsys):
        bad = {"server": {}, "dataset": {"n": 0, "domains": []}, "models": [], "prompts": []}
        p = _write_config(tmp_path, bad)
        rc = _run_dry(p, capsys)
        assert rc != 0

    def test_single_model_no_judge(self, tmp_path, capsys):
        cfg = {
            **_BASE_CFG,
            "models": [{"id": "m_a", "gguf": "org/repo-GGUF/model-a-Q4_K_M.gguf"}],
            "judge": {"enabled": False},
        }
        p = _write_config(tmp_path, cfg)
        rc = _run_dry(p, capsys)
        assert rc == 0


class TestSelectModels:
    def test_select_models_filters_and_rejects_unknown(self):
        from bench.config import ConfigError
        from bench.runner import select_models

        models = [
            {"id": "m_a", "gguf": "a.gguf"},
            {"id": "m_b", "gguf": "b.gguf"},
        ]
        assert select_models(models, None) == models
        assert [m["id"] for m in select_models(models, ["m_b"])] == ["m_b"]
        with pytest.raises(ConfigError, match="unknown --models"):
            select_models(models, ["nope"])

    def test_models_flag_restricts_dry_run(self, tmp_path, capsys):
        from bench.runner import main

        p = _write_config(tmp_path, _BASE_CFG)
        with (
            patch(
                "sys.argv",
                ["runner", "--config", str(p), "--dry-run", "--models", "m_a"],
            ),
            patch("bench.runner.write_jsonl"),
        ):
            rc = main()
        assert rc == 0
        err = capsys.readouterr().err
        # Two backends: selected model + judge.
        assert "2 backends" in err


class TestScoreAssembly:
    """Pure post-hoc rollup (no proxy/network)."""

    def _main_ok(self, mid, task_id, q):
        return {
            "model_id": mid,
            "task_id": task_id,
            "prompt_id": "plain",
            "tier": "main",
            "quality_heuristic": q,
            "failure_code": None,
            "error": None,
        }

    def _main_fail(self, mid, task_id, code):
        return {
            "model_id": mid,
            "task_id": task_id,
            "prompt_id": "plain",
            "tier": "main",
            "failure_code": code,
            "error": code,
        }

    def _floor(self, mid, task_id, passed):
        return {
            "model_id": mid,
            "task_id": task_id,
            "prompt_id": "floor",
            "tier": "dumb_model",
            "quality_heuristic": 1.0 if passed else 0.0,
            "failure_code": None,
            "error": None,
        }

    def test_assemble_partial_incomplete_and_failed(self):
        from bench.runner import assemble_model_scores

        models = [{"id": "good"}, {"id": "partial"}, {"id": "dead"}]
        # cells_total = 4 main cells per model
        records = []
        # good: 4/4 attempted, all 1.0 -> partial 1.0, complete
        for i in range(4):
            records.append(self._main_ok("good", f"t{i}", 1.0))
        # partial: 2 ok (1.0 each) + 2 timeouts -> attempted 2, S=2, partial 0.5
        records += [
            self._main_ok("partial", "t0", 1.0),
            self._main_ok("partial", "t1", 1.0),
            self._main_fail("partial", "t2", "timeout"),
            self._main_fail("partial", "t3", "timeout"),
        ]
        # dead: 4 load_failure -> attempted 0, partial 0, failed
        for i in range(4):
            records.append(self._main_fail("dead", f"t{i}", "load_failure"))
        # floor: dead passes 6/... give it 3 floor cells, 2 pass
        records += [
            self._floor("dead", "f0", True),
            self._floor("dead", "f1", True),
            self._floor("dead", "f2", False),
        ]

        scores, run_status = assemble_model_scores(models, records, cells_total=4)
        by = {s["model_id"]: s for s in scores}
        assert by["good"]["status"] == "complete"
        assert by["good"]["partial_score"] == 1.0
        assert by["partial"]["status"] == "incomplete"
        assert abs(by["partial"]["partial_score"] - 0.5) < 1e-9
        assert by["partial"]["dominant_failure_code"] == "timeout"
        assert by["dead"]["status"] == "failed"
        assert by["dead"]["partial_score"] == 0.0
        assert by["dead"]["dominant_failure_code"] == "load_failure"
        # floor score for dead = 2/3
        assert abs(by["dead"]["floor_score"] - (2 / 3)) < 1e-9
        # run-level status = worst across models = failed
        assert run_status == "failed"

    def test_floor_loader_roundtrip(self, tmp_path):
        from bench.dataset import load_floor_tasks

        p = tmp_path / "floor.jsonl"
        p.write_text(
            '{"id": "d0", "domain": "arithmetic", "user_prompt": "2+2?",'
            ' "expected": "4", "scorer": "exact"}\n'
            '{"id": "d1", "domain": "qa", "user_prompt": "capital of France?",'
            ' "expected": "Paris", "scorer": "exact", "tier": "dumb_model"}\n',
            encoding="utf-8",
        )
        tasks = load_floor_tasks(p)
        assert len(tasks) == 2
        assert all(t.tier == "dumb_model" for t in tasks)
        assert tasks[0].expected == "4"

    def test_floor_loader_absent_file_returns_empty(self, tmp_path):
        from bench.dataset import load_floor_tasks

        assert load_floor_tasks(tmp_path / "nope.jsonl") == []


class TestStoreWiring:
    """Test that runner wires to bench.store on a mocked full run."""

    def _make_config(self, tmp_path: Path, run_name: str = "test-run") -> Path:
        cfg = {
            **_BASE_CFG,
            "run": {"seed": 42, "name": run_name},
            "output": {
                "dir": str(tmp_path / "results"),
                "emit_markdown": False,
                "emit_html": False,
            },
            "judge": {"enabled": False},
            "dumb_model_tier": {"enabled": False},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    def _run_mocked(self, config_path: Path, monkeypatch, data_dir: Path) -> int:
        """Run main() with all side-effecting calls patched out."""
        from bench.runner import main

        monkeypatch.setenv("BAKEOFF_DATA_DIR", str(data_dir))

        fake_records = [
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
        ]

        with (
            patch("sys.argv", ["runner", "--config", str(config_path)]),
            patch("bench.runner.write_jsonl"),
            patch("bench.runner.engined.up"),
            patch("bench.runner.engined.down"),
            patch("bench.runner._run_model_phases", return_value=fake_records),
            patch("bench.runner.run_judge_phase", return_value=[]),
            patch(
                "bench.runner.collect_provenance",
                return_value={"warnings": [], "git": {}, "seed": 42},
            ),
            patch("bench.runner.build_model_metadata", return_value={}),
            patch("bench.runner.enrich_model_metadata", return_value={}),
            patch("bench.runner.collect_hardware_context", return_value={}),
        ):
            return main()

    def test_store_record_written_after_run(self, tmp_path, monkeypatch):
        """Store file must exist at runs/<run_id>.json after a successful run."""
        from bench import store

        data_dir = tmp_path / "data"
        config_path = self._make_config(tmp_path, run_name="test-run-42")
        rc = self._run_mocked(config_path, monkeypatch, data_dir)

        assert rc == 0
        monkeypatch.setenv("BAKEOFF_DATA_DIR", str(data_dir))
        run_ids = store.list_runs()
        assert len(run_ids) == 1
        record = store.read_record("runs", run_ids[0])
        assert record["run_id"] == run_ids[0]
        assert record["run_id"] != "test-run-42"
        assert "records" in record

    def test_queue_completed_record_written(self, tmp_path, monkeypatch):
        """run_queue/completed/<qid>.json must exist; pending must be gone."""
        data_dir = tmp_path / "data"
        config_path = self._make_config(tmp_path, run_name="q-test-run")
        rc = self._run_mocked(config_path, monkeypatch, data_dir)

        assert rc == 0
        completed_dir = data_dir / "run_queue" / "completed"
        pending_dir = data_dir / "run_queue" / "pending"
        completed = list(completed_dir.glob("*.json")) if completed_dir.exists() else []
        pending = list(pending_dir.glob("*.json")) if pending_dir.exists() else []
        assert len(completed) == 1, "expected one completed queue record"
        assert len(pending) == 0, "pending record should have been moved to completed"

    def test_list_runs_returns_stored_run(self, tmp_path, monkeypatch):
        """list_runs() must return the run ID after a successful run."""
        from bench import store

        data_dir = tmp_path / "data"
        config_path = self._make_config(tmp_path, run_name="listed-run")
        self._run_mocked(config_path, monkeypatch, data_dir)

        monkeypatch.setenv("BAKEOFF_DATA_DIR", str(data_dir))
        run_ids = store.list_runs()
        assert len(run_ids) == 1
        assert run_ids[0] != "listed-run"

    def test_run_ids_are_unique_across_runs(self, tmp_path, monkeypatch):
        """Two runs with the same config name must not share a run_id."""
        from bench import store

        data_dir = tmp_path / "data"
        config_path = self._make_config(tmp_path, run_name="same-name")
        assert self._run_mocked(config_path, monkeypatch, data_dir) == 0
        assert self._run_mocked(config_path, monkeypatch, data_dir) == 0

        monkeypatch.setenv("BAKEOFF_DATA_DIR", str(data_dir))
        run_ids = store.list_runs()
        assert len(set(run_ids)) == 2


def test_resume_seed_mismatch_exits_without_reuse(tmp_path, monkeypatch):
    from bench.runner import main

    cfg = {
        **_BASE_CFG,
        "run": {"seed": 2, "name": "new-run"},
        "output": {
            "dir": str(tmp_path / "results"),
            "emit_markdown": False,
            "emit_html": False,
        },
        "judge": {"enabled": False},
        "dumb_model_tier": {"enabled": False},
    }
    config_path = _write_config(tmp_path, cfg)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "tasks": [{"id": "t0000"}],
                "records": [
                    {
                        "task_id": "t0000",
                        "prompt_id": "plain",
                        "model_id": "m_a",
                        "text": "old-answer",
                    }
                ],
                "config": {
                    "models": cfg["models"],
                    "prompts": cfg["prompts"],
                },
                "provenance": {"seed": 1},
                "run_id": "old-run",
            }
        )
    )
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path / "data"))
    with (
        patch(
            "sys.argv",
            ["runner", "--config", str(config_path), "--resume-from", str(prior_path)],
        ),
        patch("bench.runner.write_jsonl"),
        patch("bench.runner.engined.up") as up,
        patch("bench.runner.engined.down"),
        patch("bench.runner._run_model_phases", return_value=[]) as phases,
        patch("bench.runner.run_judge_phase", return_value=[]),
        patch(
            "bench.runner.collect_provenance", return_value={"warnings": [], "git": {}, "seed": 2}
        ),
        patch("bench.runner.build_model_metadata", return_value={}),
        patch("bench.runner.enrich_model_metadata", return_value={}),
        patch("bench.runner.collect_hardware_context", return_value={}),
    ):
        rc = main()
    assert rc != 0
    up.assert_not_called()
    phases.assert_not_called()
    assert list((tmp_path / "results").glob("run-*.json")) == []


def test_crash_after_first_model_keeps_its_records(tmp_path, monkeypatch):
    from bench.runner import main

    cfg = {
        **_BASE_CFG,
        "run": {"seed": 42, "name": "partial-run", "warmup": False},
        "output": {
            "dir": str(tmp_path / "results"),
            "emit_markdown": False,
            "emit_html": False,
        },
        "judge": {"enabled": False},
        "dumb_model_tier": {"enabled": False},
    }
    config_path = _write_config(tmp_path, cfg)
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path / "data"))

    def fake_phase(model_cfg, *args, **kwargs):
        if str(model_cfg["id"]) == "m_b":
            raise RuntimeError("model b exploded")
        return [
            {
                "model_id": "m_a",
                "task_id": "t0000",
                "prompt_id": "plain",
                "text": "from-a",
                "tier": "main",
                "failure_code": None,
                "error": None,
            }
        ]

    with (
        patch("sys.argv", ["runner", "--config", str(config_path)]),
        patch("bench.runner.write_jsonl"),
        patch("bench.runner.engined.up"),
        patch("bench.runner.engined.down"),
        patch("bench.runner.run_model_phase", side_effect=fake_phase),
        patch("bench.runner.collect_hardware_context", return_value={}),
        pytest.raises(RuntimeError, match="model b exploded"),
    ):
        main()

    files = list((tmp_path / "results").glob("run-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert [r["model_id"] for r in payload["records"]] == ["m_a"]
    assert payload["judgements"] == []


def test_refusal_body_sets_failure_code(monkeypatch):
    from bench.clients import ChatResult
    from bench.dataset import Task
    from bench.runner import run_model_phase

    res = ChatResult(
        text="I cannot help with that request.",
        prompt_tokens=1,
        completion_tokens=1,
        latency_s=0.1,
        ttft_s=0.01,
        tokens_per_sec=10.0,
        raw={},
    )
    monkeypatch.setattr(
        "bench.runner.call_one",
        lambda *a, **k: (res, None, None, None, None, None),
    )
    recs = run_model_phase(
        {"id": "m_a"},
        [Task(id="t0", domain="qa", user_prompt="hi")],
        [{"id": "plain", "system": ""}],
        "http://localhost",
        {"enabled": False},
        5.0,
        warmup=False,
    )
    assert recs[0]["failure_code"] == "refusal"


def test_portable_result_write_is_atomic(tmp_path, monkeypatch):
    from bench.store import atomic_write_json

    out = tmp_path / "run-x.json"
    out.write_text('{"keep": true}')
    original = out.read_text()

    def boom(obj, f, **kwargs):
        f.write("{truncated")
        raise RuntimeError("midway")

    monkeypatch.setattr("bench.store.json.dump", boom)
    with pytest.raises(RuntimeError, match="midway"):
        atomic_write_json(out, {"new": True})
    assert out.read_text() == original
    leftover = [p for p in tmp_path.glob("*.tmp") if p.exists()]
    assert leftover == []
