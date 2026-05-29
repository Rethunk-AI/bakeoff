"""Unit tests for bench.publish result bundle workflow."""

from __future__ import annotations

import json

from bench.publish import (
    _SCORE_SUMMARY_KEYS,
    SCHEMA_VERSION,
    main,
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
        "records": [
            {
                "task_id": "t1",
                "prompt_id": "plain",
                "model_id": "m_a",
                "text": "Answer",
                "wall_clock_seconds": 1.0,
                "seconds_to_first_token": 0.1,
                "tokens_per_second": 12.0,
                "energy_wh": None,
                "cost_usd": None,
                "quality_heuristic": 1.0,
                "error": None,
            }
        ],
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


# --- §4 manifest schema additions ---


def _payload_with_scores():
    """Base payload extended with run_status and model_scores."""
    p = _payload()
    p["run_status"] = "incomplete"
    p["model_scores"] = [
        {
            "model_id": "m_a",
            "status": "incomplete",
            "cells_total": 10,
            "cells_attempted": 6,
            "cells_failed": 2,
            "completeness": 0.6,
            "partial_score": 0.54,
            "floor_score": 0.67,
            "dominant_failure_code": "timeout",
        }
    ]
    return p


def test_manifest_projection_contains_only_summary_keys(tmp_path):
    """Manifest model_scores_summary must contain exactly the 5 projected keys."""
    bundle = package_result(_write_payload(tmp_path, _payload_with_scores()), tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())

    assert manifest["run_status"] == "incomplete"
    summary = manifest["model_scores_summary"]
    assert isinstance(summary, list) and len(summary) == 1
    entry = summary[0]
    # Only the 5 projected keys — no extras like cells_total, completeness, etc.
    assert set(entry.keys()) == set(_SCORE_SUMMARY_KEYS)
    assert entry["model_id"] == "m_a"
    assert entry["status"] == "incomplete"
    assert entry["partial_score"] == 0.54
    assert entry["floor_score"] == 0.67
    assert entry["dominant_failure_code"] == "timeout"


def test_backward_compat_old_payload_validates_clean(tmp_path):
    """Payloads without run_status / model_scores must validate without errors."""
    # validate_result_payload: no errors on base payload
    assert validate_result_payload(_payload()) == []

    # package_result + validate_bundle: old bundles validate clean
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    assert validate_bundle(bundle) == []

    # manifest gets run_status=None and model_scores_summary=None for old payloads
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["run_status"] is None
    assert manifest["model_scores_summary"] is None


def test_strict_warns_not_errors_on_absent_fields(tmp_path, capsys):
    """--strict on a payload without new fields prints warnings but exits 0."""
    result = _write_payload(tmp_path)
    rc = main(["validate", "--strict", str(result)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "run_status" in captured.err
    assert "model_scores" in captured.err
