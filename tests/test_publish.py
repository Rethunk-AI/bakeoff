"""Unit tests for bench.publish result bundle workflow."""
from __future__ import annotations

import json

from bench.publish import (
    SCHEMA_VERSION,
    package_result,
    submit_bundle,
    validate_bundle,
    validate_result_payload,
)


def _payload():
    return {
        "run_id": "test-run",
        "timestamp": "20260427-010000",
        "config": {
            "models": [{"id": "m_a", "gguf": "org/repo/model-Q4_K_M.gguf"}],
            "prompts": [{"id": "plain", "system": "Answer."}],
            "judge": {"enabled": True, "mode": "pairwise_all"},
        },
        "provenance": {
            "config_hash": "abc123",
            "seed": 42,
            "git": {"sha": "deadbee", "branch": "main", "dirty": False},
        },
        "model_metadata": [
            {"id": "m_a", "gguf": "org/repo/model-Q4_K_M.gguf", "repo_id": "org/repo"},
        ],
        "tasks": [{"id": "t1", "domain": "qa", "user_prompt": "Question?"}],
        "records": [{
            "task_id": "t1",
            "prompt_id": "plain",
            "model_id": "m_a",
            "text": "Answer",
            "latency_s": 1.0,
            "ttft_s": 0.1,
            "tokens_per_sec": 12.0,
            "energy_wh": None,
            "cost_usd": None,
            "quality_heuristic": 1.0,
            "error": None,
        }],
        "judgements": [],
    }


def _write_payload(tmp_path, payload=None):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload or _payload()))
    return path


def test_validate_result_payload_accepts_current_shape():
    assert validate_result_payload(_payload()) == []


def test_validate_result_payload_requires_provenance():
    payload = _payload()
    payload["provenance"].pop("config_hash")
    errors = validate_result_payload(payload)
    assert any("config_hash" in e for e in errors)


def test_package_result_writes_bundle(tmp_path):
    result = _write_payload(tmp_path)
    bundle = package_result(result, tmp_path / "bundle")
    assert (bundle / "result.json").is_file()
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "summary.md").is_file()
    assert (bundle / "dashboard.html").is_file()
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert validate_bundle(bundle) == []


def test_validate_bundle_detects_tampered_result(tmp_path):
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    result = bundle / "result.json"
    result.write_text(result.read_text().replace("test-run", "evil-run"))
    errors = validate_bundle(bundle)
    assert any("sha256 mismatch" in e for e in errors)


def test_submit_bundle_dry_run_returns_target(tmp_path, capsys):
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    target = submit_bundle(bundle, checkout=tmp_path / "results", dry_run=True)
    assert target == tmp_path / "results" / "submissions" / "test-run"
    assert "would submit" in capsys.readouterr().out
