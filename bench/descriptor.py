"""Model descriptor reader, validator, and persister for the bakeoff harness.

SCOPE: handles manual/admin/seed JSON descriptors only. The full pipeline
(URL-submit → upstream-metadata-pull → embedded-GGUF-metadata cross-check)
is explicitly DEFERRED to P2 per bakeoff#15 design agreement. The only
ingestion path handled here is hand-crafted or admin-supplied JSON files.

Descriptor JSON layout mirrors the ``models`` table in schema/schema.sql
with embedded ``creator`` (1:1) and ``sources`` array (model_sources, 1:many):

.. code-block:: json

    {
      "schema_version": 1,
      "name": "Qwen3-8B-Q4_K_M",
      "creator": {
        "name": "qwen",
        "display_name": "Qwen Team",
        "homepage": "https://huggingface.co/Qwen",
        "service_identifiers": {"huggingface": "Qwen"},
        "provisional": false
      },
      "model_hash": "sha256:abcdef...",
      "parameter_count_b": 8.0,
      "active_parameter_count_b": 8.0,
      "architecture": "Dense",
      "file_format": "GGUF",
      "quantization": "q4_k_m",
      "context_length_default": 8192,
      "context_length_min": null,
      "context_length_max": 131072,
      "model_source_mtime": null,
      "model_source_size": null,
      "release_date": null,
      "version": null,
      "description": null,
      "predecessor_model_id": null,
      "provisional": false,
      "sources": [
        {
          "source_type": "huggingface",
          "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF",
          "source_metadata": {"hf_repo": "Qwen/Qwen3-8B-GGUF"}
        }
      ]
    }

Validation rules (mirroring bench.config error-collection style):
- ``schema_version`` must be present and equal to SCHEMA_VERSION (int 1).
  load_descriptor() raises DescriptorError immediately on missing or
  unsupported schema_version — this gate exists explicitly for forward migration.
- ``name`` is required.
- At least one entry in ``sources`` is required.
- Either ``model_hash`` OR (a source with a url + parameter_count_b + model_source_size)
  must be present so a deterministic UUID can be derived.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench import store

SCHEMA_VERSION = 1


class DescriptorError(ValueError):
    """Raised when a descriptor cannot be loaded or has an unsupported schema_version."""


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------


def load_descriptor(path: Path) -> dict[str, Any]:
    """Parse and validate a descriptor JSON file. Return the validated dict.

    Raises:
        DescriptorError: if the file is missing, not valid JSON, or has a
            missing/unsupported ``schema_version``. This gate exists so
            forward-migration errors are always visible at load time.
    """
    try:
        with path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        raise DescriptorError(f"descriptor file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise DescriptorError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise DescriptorError(f"{path} must contain a JSON object")

    # schema_version gate — explicit fail, not a validation issue list.
    sv = data.get("schema_version")
    if sv is None:
        raise DescriptorError(
            f"{path}: 'schema_version' is required (expected {SCHEMA_VERSION})"
        )
    if sv != SCHEMA_VERSION:
        raise DescriptorError(
            f"{path}: unsupported schema_version {sv!r} (expected {SCHEMA_VERSION})"
        )

    return data


def validate_descriptor(data: dict[str, Any]) -> list[str]:
    """Validate descriptor dict fields. Return a list of issue strings.

    An empty list means valid. Mirrors the bench.config validation style
    (aggregate issues rather than raising on first error).
    """
    issues: list[str] = []

    def err(msg: str) -> None:
        issues.append(msg)

    # name is required.
    if not data.get("name"):
        err("name: required")

    # sources: must have at least one entry.
    sources = data.get("sources")
    if not isinstance(sources, list) or len(sources) == 0:
        err("sources: must be a non-empty list")
    else:
        for i, s in enumerate(sources):
            if not isinstance(s, dict):
                err(f"sources[{i}]: must be an object")
            else:
                if not s.get("source_type"):
                    err(f"sources[{i}].source_type: required")
                if not s.get("url"):
                    err(f"sources[{i}].url: required")

    # UUID derivation: need model_hash OR (source url + parameter_count_b + model_source_size).
    model_hash = data.get("model_hash")
    if not model_hash:
        has_url = bool(
            isinstance(sources, list)
            and sources
            and isinstance(sources[0], dict)
            and sources[0].get("url")
        )
        has_params = data.get("parameter_count_b") is not None
        has_size = data.get("model_source_size") is not None
        if not (has_url and has_params and has_size):
            err(
                "uuid derivation: either 'model_hash' or "
                "('sources[0].url' + 'parameter_count_b' + 'model_source_size') is required"
            )

    return issues


# ---------------------------------------------------------------------------
# UUID derivation
# ---------------------------------------------------------------------------


def descriptor_uuid(data: dict[str, Any]) -> str:
    """Derive the model UUID from a descriptor.

    If ``model_hash`` is present → ``store.model_uuid(model_hash)``.
    Otherwise → ``store.provisional_model_uuid(url, params, size)`` using
    the first source URL, ``parameter_count_b``, and ``model_source_size``.
    """
    model_hash = data.get("model_hash")
    if model_hash:
        return store.model_uuid(str(model_hash))

    sources = data.get("sources") or []
    url = sources[0]["url"] if sources and isinstance(sources[0], dict) else ""
    params = data.get("parameter_count_b")
    size = data.get("model_source_size")
    return store.provisional_model_uuid(url, params, size)


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


def save_descriptor(data: dict[str, Any]) -> str:
    """Stamp UUID and audit fields on *data*, then persist to the store.

    Returns the UUID string that was assigned.
    Raises DescriptorError if validation fails before saving.
    """
    issues = validate_descriptor(data)
    if issues:
        raise DescriptorError("invalid descriptor:\n- " + "\n- ".join(issues))

    uuid = descriptor_uuid(data)
    payload = dict(data)
    payload["uuid"] = uuid
    store.write_record("models", uuid, payload)
    return uuid
