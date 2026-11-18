"""Unit tests for bench.publish result bundle workflow."""

from __future__ import annotations

import hashlib
import json

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from bench.publish import (
    _SCORE_SUMMARY_KEYS,
    SCHEMA_VERSION,
    PublishError,
    main,
    package_result,
    submit_bundle,
    validate_bundle,
    validate_path,
    validate_result_payload,
)
from bench.signing import generate_keypair, sign_result


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


def test_validate_bundle_rejects_symlink_member(tmp_path):
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    secret = tmp_path / "secret.txt"
    secret.write_text("leaked-secret\n")
    dash = bundle / "dashboard.html"
    dash.unlink()
    dash.symlink_to(secret)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["dashboard.html"]["sha256"] = hashlib.sha256(secret.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    errors = validate_bundle(bundle)
    assert any("symlink" in e or "unsafe" in e for e in errors)


def test_validate_bundle_rejects_absolute_manifest_path(tmp_path):
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    secret = tmp_path / "outside.txt"
    secret.write_text("outside\n")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][str(secret)] = {"sha256": hashlib.sha256(secret.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    errors = validate_bundle(bundle)
    assert any("unsafe" in e for e in errors)


def test_validate_bundle_rejects_unlisted_symlink(tmp_path):
    """A symlink the manifest never names must not reach the submission."""
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    secret = tmp_path / "secret.txt"
    secret.write_text("leaked-secret\n")
    (bundle / "innocent.txt").symlink_to(secret)

    errors = validate_bundle(bundle)
    assert any("innocent.txt" in e and "symlink" in e for e in errors)

    dest = tmp_path / "results" / "submissions" / "test-run"
    with pytest.raises(PublishError):
        submit_bundle(bundle, checkout=tmp_path / "results")
    assert not dest.exists()


def test_validate_bundle_rejects_unlisted_entry(tmp_path):
    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    (bundle / "extra.txt").write_text("not in manifest\n")
    errors = validate_bundle(bundle)
    assert any("extra.txt" in e and "not listed in manifest" in e for e in errors)


def test_malformed_result_fields_are_named():
    missing = validate_result_payload({})
    assert any("run_id" in e for e in missing)
    assert any("timestamp" in e for e in missing)
    assert any(e == "config: required object" for e in missing)
    assert any(e == "provenance: required object" for e in missing)
    assert any("tasks: required non-empty list" in e for e in missing)

    nested = validate_result_payload(
        {
            "run_id": "r",
            "timestamp": "t",
            "config": {
                "models": ["nope", {"id": "", "gguf": ""}],
                "prompts": [{}, {"id": "p"}],
            },
            "provenance": {"git": "not-an-object"},
            "tasks": [{}, {"id": "t1"}],
            "records": ["nope", {}],
            "model_metadata": ["nope", {}],
            "run_status": "maybe",
            "model_scores": {"not": "a list"},
        }
    )
    joined = "\n".join(nested)
    assert "provenance.config_hash" in joined
    assert "provenance.seed" in joined
    assert "provenance.git.sha" in joined
    assert "config.models[0]: required object" in joined
    assert "config.models[1].id" in joined
    assert "config.models[1].gguf" in joined
    assert "config.prompts[0].id" in joined
    assert "tasks[0].id" in joined
    assert "records[0]: required object" in joined
    assert "records[1].task_id" in joined
    assert "model_metadata[0]: required object" in joined
    assert "model_metadata[1].id" in joined
    assert "model_metadata[1].gguf" in joined
    assert "run_status:" in joined
    assert "model_scores: must be a list" in joined


def test_unreadable_result_files_are_rejected(tmp_path):
    assert any("not found" in e for e in validate_path(tmp_path / "missing.json"))
    (tmp_path / "bad.json").write_text("{nope")
    assert any("invalid JSON" in e for e in validate_path(tmp_path / "bad.json"))
    (tmp_path / "arr.json").write_text("[]")
    assert any("JSON object" in e for e in validate_path(tmp_path / "arr.json"))


def test_signed_envelope_needs_a_matching_pem_public_key(tmp_path):
    priv, pub_b64 = generate_keypair()
    other_priv, other_pub = generate_keypair()
    envelope = sign_result(_payload(), priv, "runner-1")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(envelope))

    assert any("no public key" in e for e in validate_path(path))
    assert any("signature verification failed" in e for e in validate_path(path, other_pub))

    pem = priv.public_key().public_bytes(
        encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
    )
    key_path = tmp_path / "pub.pem"
    key_path.write_bytes(pem)
    assert main(["validate", str(path), "--public-key", str(key_path)]) == 0

    b64_path = tmp_path / "pub.b64"
    b64_path.write_text(pub_b64 + "\n")
    assert main(["validate", str(path), "--public-key", str(b64_path)]) == 0

    wrong_pem = other_priv.public_key().public_bytes(
        encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
    )
    wrong_path = tmp_path / "wrong.pem"
    wrong_path.write_bytes(wrong_pem)
    assert main(["validate", str(path), "--public-key", str(wrong_path)]) == 1


def test_bundle_rejects_incomplete_manifest_and_required_unsigned_signature(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert any("missing" in e for e in validate_bundle(empty))

    bundle = package_result(_write_payload(tmp_path), tmp_path / "ok-bundle")
    assert validate_path(bundle) == []

    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle")
    (bundle / "result.json").unlink()
    assert any("missing" in e for e in validate_bundle(bundle))

    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle2")
    (bundle / "manifest.json").write_text("[]\n")
    assert any("JSON object" in e for e in validate_bundle(bundle))

    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle3")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    assert any("manifest.files" in e for e in validate_bundle(bundle))

    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle4")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["dashboard.html"] = {}
    del manifest["schema_version"]
    manifest["run_id"] = ""
    manifest["created_at"] = ""
    manifest["signature"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n")
    errors = validate_bundle(bundle)
    joined = "\n".join(errors)
    assert "schema_version" in joined
    assert "manifest.run_id" in joined
    assert "manifest.created_at" in joined
    assert "sha256" in joined
    assert "manifest.signature" in joined

    bundle = package_result(_write_payload(tmp_path), tmp_path / "bundle5")
    (bundle / "dashboard.html").unlink()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["signature"] = {"required": True}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    errors = validate_bundle(bundle)
    joined = "\n".join(errors)
    assert "dashboard.html" in joined and "missing" in joined
    assert "signature.path" in joined


def test_package_refuses_invalid_payload_and_existing_output(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_id": "x"}))
    with pytest.raises(PublishError, match="invalid result"):
        package_result(bad, tmp_path / "out")

    result = _write_payload(tmp_path)
    payload = json.loads(result.read_text())
    payload["run_id"] = "???"
    result.write_text(json.dumps(payload))
    first = package_result(result)
    assert first.name == "run"
    with pytest.raises(PublishError, match="exists"):
        package_result(result)
    assert package_result(result, force=True) == first
