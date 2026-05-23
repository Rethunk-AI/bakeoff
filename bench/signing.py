"""Ed25519 sign and verify for bakeoff result envelopes.

Envelope format
---------------
{
    "result": { <all benchmark output fields> },
    "sig": {
        "sha256":    "<hex SHA256 of canonical(result)>",
        "signature": "<base64 Ed25519 signature over sha256_bytes>",
        "runner_id": "<runner identity string>",
        "signed_at": "<ISO8601 UTC>"
    }
}

Canonical form: json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
Signature input: raw SHA256 bytes (not the hex string).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class SigningError(ValueError):
    """Raised when signature verification fails."""


def canonical_json(data: dict[str, Any]) -> bytes:
    """Return canonical (sorted-keys, no-whitespace) JSON bytes for *data*."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def generate_keypair() -> tuple[Ed25519PrivateKey, str]:
    """Generate a new Ed25519 keypair.

    Returns
    -------
    (private_key, public_key_b64)
        *private_key* is a live key object ready for signing.
        *public_key_b64* is the base64-encoded raw 32-byte public key,
        suitable for storing in the ``runners`` table.
    """
    private_key = Ed25519PrivateKey.generate()
    public_key_b64 = _encode_public_key(private_key.public_key())
    return private_key, public_key_b64


def _encode_public_key(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode()


def sign_result(
    result: dict[str, Any],
    private_key: Ed25519PrivateKey,
    runner_id: str,
) -> dict[str, Any]:
    """Wrap *result* in a signed envelope.

    Parameters
    ----------
    result:
        The raw benchmark payload dict (not yet wrapped).
    private_key:
        Ed25519 private key used for signing.
    runner_id:
        Runner identity string stored in ``sig.runner_id``.

    Returns
    -------
    dict
        Signed envelope with top-level keys ``result`` and ``sig``.
    """
    canonical = canonical_json(result)
    sha256_bytes = hashlib.sha256(canonical).digest()
    sha256_hex = sha256_bytes.hex()
    signature_bytes = private_key.sign(sha256_bytes)
    signature_b64 = base64.b64encode(signature_bytes).decode()
    signed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "result": result,
        "sig": {
            "sha256": sha256_hex,
            "signature": signature_b64,
            "runner_id": runner_id,
            "signed_at": signed_at,
        },
    }


def verify_result(envelope: dict[str, Any], public_key_b64: str) -> dict[str, Any]:
    """Verify a signed result envelope and return the unwrapped result dict.

    Parameters
    ----------
    envelope:
        Dict with top-level keys ``result`` and ``sig``.
    public_key_b64:
        Base64-encoded raw 32-byte Ed25519 public key.

    Returns
    -------
    dict
        The ``result`` dict from inside the envelope.

    Raises
    ------
    SigningError
        If the envelope is missing required keys, the SHA256 does not match,
        or the signature is invalid.
    """
    if "result" not in envelope or "sig" not in envelope:
        raise SigningError("envelope missing required keys: 'result' and/or 'sig'")

    result = envelope["result"]
    sig = envelope["sig"]

    # Recompute canonical hash of result
    canonical = canonical_json(result)
    sha256_bytes = hashlib.sha256(canonical).digest()
    sha256_hex = sha256_bytes.hex()

    # Compare recorded hash
    recorded_hex = sig.get("sha256", "")
    if sha256_hex != recorded_hex:
        raise SigningError(
            f"SHA256 mismatch: computed {sha256_hex!r}, envelope has {recorded_hex!r}"
        )

    # Decode signature and public key
    try:
        signature_bytes = base64.b64decode(sig["signature"])
    except Exception as exc:
        raise SigningError(f"invalid base64 signature: {exc}") from exc

    try:
        raw_pub = base64.b64decode(public_key_b64)
        public_key = Ed25519PublicKey.from_public_bytes(raw_pub)
    except Exception as exc:
        raise SigningError(f"invalid public key: {exc}") from exc

    # Verify signature
    try:
        public_key.verify(signature_bytes, sha256_bytes)
    except InvalidSignature as exc:
        raise SigningError("Ed25519 signature verification failed") from exc

    return result


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM (PKCS8) file."""
    pem_bytes = path.read_bytes()
    key = serialization.load_pem_private_key(pem_bytes, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError(f"{path} does not contain an Ed25519 private key")
    return key


def save_private_key(key: Ed25519PrivateKey, path: Path) -> None:
    """Save *key* to *path* as an unencrypted PEM (PKCS8) file."""
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem_bytes)
