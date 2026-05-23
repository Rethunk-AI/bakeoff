"""Unit tests for bench.signing — Ed25519 sign/verify module."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from bench.signing import (
    SigningError,
    canonical_json,
    generate_keypair,
    sign_result,
    verify_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESULT: dict = {
    "run_id": "test-run-001",
    "timestamp": "20260522-120000",
    "config": {"models": [], "prompts": []},
    "records": [],
}


# ---------------------------------------------------------------------------
# canonical_json
# ---------------------------------------------------------------------------


def test_canonical_json_stable_across_insertion_order() -> None:
    """canonical_json must produce identical output regardless of dict ordering."""
    d1 = {"b": 2, "a": 1, "c": [3, 1, 2]}
    d2 = {"c": [3, 1, 2], "a": 1, "b": 2}
    assert canonical_json(d1) == canonical_json(d2)


def test_canonical_json_sorted_keys() -> None:
    """Keys in canonical output must be sorted."""
    data = {"z": 1, "a": 2, "m": 3}
    parsed = json.loads(canonical_json(data))
    assert list(parsed.keys()) == ["a", "m", "z"]


def test_canonical_json_no_whitespace() -> None:
    """canonical_json must use compact separators (no spaces)."""
    data = {"key": "value"}
    result = canonical_json(data)
    assert b" " not in result


def test_canonical_json_returns_bytes() -> None:
    assert isinstance(canonical_json({"x": 1}), bytes)


# ---------------------------------------------------------------------------
# generate_keypair
# ---------------------------------------------------------------------------


def test_generate_keypair_returns_valid_objects() -> None:
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    private_key, public_key_b64 = generate_keypair()
    assert isinstance(private_key, Ed25519PrivateKey)
    # public_key_b64 must be valid base64 decoding to 32 bytes (Ed25519 raw pubkey)
    raw = base64.b64decode(public_key_b64)
    assert len(raw) == 32
    # Reconstructing the public key from the returned b64 must work
    pub = private_key.public_key()
    assert isinstance(pub, Ed25519PublicKey)


def test_generate_keypair_different_each_call() -> None:
    _, pub1 = generate_keypair()
    _, pub2 = generate_keypair()
    assert pub1 != pub2


# ---------------------------------------------------------------------------
# sign_result
# ---------------------------------------------------------------------------


def test_sign_result_envelope_structure() -> None:
    private_key, _ = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "test-runner")
    assert "result" in envelope
    assert "sig" in envelope
    sig = envelope["sig"]
    assert "sha256" in sig
    assert "signature" in sig
    assert "runner_id" in sig
    assert "signed_at" in sig
    assert sig["runner_id"] == "test-runner"


def test_sign_result_sha256_is_hex() -> None:
    import hashlib

    private_key, _ = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    sha256_hex = envelope["sig"]["sha256"]
    # Must be valid 64-char hex
    assert len(sha256_hex) == 64
    int(sha256_hex, 16)  # raises ValueError if not hex
    # Must match canonical_json of the result
    expected = hashlib.sha256(canonical_json(SAMPLE_RESULT)).hexdigest()
    assert sha256_hex == expected


def test_sign_result_result_field_is_original() -> None:
    private_key, _ = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    assert envelope["result"] == SAMPLE_RESULT


def test_sign_result_signed_at_format() -> None:
    private_key, _ = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    # Should parse as ISO8601 UTC
    dt = datetime.strptime(envelope["sig"]["signed_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert dt is not None


# ---------------------------------------------------------------------------
# verify_result — success cases
# ---------------------------------------------------------------------------


def test_verify_result_success() -> None:
    private_key, public_key_b64 = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    result = verify_result(envelope, public_key_b64)
    assert result == SAMPLE_RESULT


def test_round_trip_sign_verify_returns_original() -> None:
    data = {"tasks": [1, 2, 3], "run_id": "abc", "nested": {"x": True}}
    private_key, public_key_b64 = generate_keypair()
    envelope = sign_result(data, private_key, "my-runner")
    recovered = verify_result(envelope, public_key_b64)
    assert recovered == data


# ---------------------------------------------------------------------------
# verify_result — failure cases
# ---------------------------------------------------------------------------


def test_verify_result_raises_on_missing_result_key() -> None:
    private_key, public_key_b64 = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    bad = {"sig": envelope["sig"]}  # missing "result"
    with pytest.raises(SigningError, match="missing required keys"):
        verify_result(bad, public_key_b64)


def test_verify_result_raises_on_missing_sig_key() -> None:
    private_key, public_key_b64 = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    bad = {"result": envelope["result"]}  # missing "sig"
    with pytest.raises(SigningError, match="missing required keys"):
        verify_result(bad, public_key_b64)


def test_verify_result_raises_on_tampered_result() -> None:
    private_key, public_key_b64 = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    # Tamper with the result after signing
    tampered = dict(envelope)
    tampered["result"] = dict(envelope["result"])
    tampered["result"]["injected_field"] = "evil"
    with pytest.raises(SigningError, match="SHA256 mismatch"):
        verify_result(tampered, public_key_b64)


def test_verify_result_raises_on_wrong_public_key() -> None:
    private_key, _pub = generate_keypair()
    envelope = sign_result(SAMPLE_RESULT, private_key, "runner")
    # Generate a different keypair — wrong public key
    _, wrong_pub = generate_keypair()
    with pytest.raises(SigningError, match=r"[Ss]ignature"):
        verify_result(envelope, wrong_pub)


def test_verify_result_raises_on_empty_envelope() -> None:
    _, public_key_b64 = generate_keypair()
    with pytest.raises(SigningError, match="missing required keys"):
        verify_result({}, public_key_b64)
