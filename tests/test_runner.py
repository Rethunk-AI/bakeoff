"""Smoke tests for bench.runner dry-run path.

These tests exercise the full dry-run code path (config load → validation →
dataset generation → proxy config generation) without starting a proxy or
making any network calls. They serve as CI fixtures that catch regressions
in non-default config branches (alternate judge modes, judge disabled, etc.).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

# Minimal valid config shared by multiple tests.
_BASE_CFG = {
    "run": {"seed": 42},
    "server": {
        "image": "ghcr.io/ggml-org/llama.cpp:server-vulkan",
        "ctx": 4096,
        "ngl": 99,
        "models_dir": "/tmp/models",
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
    with patch("sys.argv", ["runner", "--config", str(config_path), "--dry-run"]), \
         patch("bench.runner.write_jsonl"):
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
        cfg = {**_BASE_CFG, "judge": {
            "enabled": True, "mode": "scored",
            "gguf": "org/repo-GGUF/judge-Q4_K_M.gguf", "ctx": 4096,
        }}
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
