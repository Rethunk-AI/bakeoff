"""Tests for bench.descriptor — model descriptor reader/validator/persister.

Coverage strategy: one happy-path roundtrip covering load + validate + save,
then critical edges: missing schema_version, wrong schema_version, UUID
derivation (hash vs provisional).
"""

from __future__ import annotations

import json

import pytest

from bench import descriptor, store


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point BAKEOFF_DATA_DIR at a per-test temp directory."""
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures — minimal valid descriptor data
# ---------------------------------------------------------------------------

VALID_DESCRIPTOR: dict = {
    "schema_version": 1,
    "name": "TestModel-Q4_K_M",
    "creator": {
        "name": "test_org",
        "display_name": "Test Org",
        "homepage": "https://example.com",
        "provisional": False,
    },
    "model_hash": "sha256:deadbeef1234",
    "parameter_count_b": 7.0,
    "active_parameter_count_b": 7.0,
    "architecture": "Dense",
    "file_format": "GGUF",
    "quantization": "q4_k_m",
    "context_length_default": 4096,
    "context_length_min": None,
    "context_length_max": 32768,
    "model_source_mtime": None,
    "model_source_size": 4_000_000_000,
    "release_date": None,
    "version": None,
    "description": None,
    "predecessor_model_id": None,
    "provisional": False,
    "sources": [
        {
            "source_type": "huggingface",
            "url": "https://huggingface.co/test_org/TestModel-GGUF",
            "source_metadata": {"hf_repo": "test_org/TestModel-GGUF"},
        }
    ],
}

PROVISIONAL_DESCRIPTOR: dict = {
    "schema_version": 1,
    "name": "ProvisionalModel-Q4_K_M",
    "creator": {"name": "anon", "provisional": True},
    # No model_hash — UUID derived from url + params + size.
    "parameter_count_b": 3.0,
    "model_source_size": 2_000_000_000,
    "sources": [
        {
            "source_type": "direct_url",
            "url": "https://cdn.example.com/model.gguf",
        }
    ],
}


# ---------------------------------------------------------------------------
# Happy path: load + validate + save roundtrip
# ---------------------------------------------------------------------------


def test_load_descriptor_happy(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(VALID_DESCRIPTOR))

    data = descriptor.load_descriptor(path)
    assert data["name"] == "TestModel-Q4_K_M"
    assert data["schema_version"] == 1


def test_validate_descriptor_happy():
    issues = descriptor.validate_descriptor(VALID_DESCRIPTOR)
    assert issues == []


def test_save_descriptor_roundtrip():
    uuid = descriptor.save_descriptor(VALID_DESCRIPTOR)
    # UUID must match the hash-based derivation.
    assert uuid == store.model_uuid("sha256:deadbeef1234")
    # Record must be retrievable.
    record = store.read_record("models", uuid)
    assert record["name"] == "TestModel-Q4_K_M"
    assert "uuid" in record
    assert record["uuid"] == uuid
    assert "created_at" in record
    assert "updated_at" in record


# ---------------------------------------------------------------------------
# schema_version gate
# ---------------------------------------------------------------------------


def test_load_descriptor_missing_schema_version(tmp_path):
    bad = {"name": "NoVersion", "sources": []}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))

    with pytest.raises(descriptor.DescriptorError, match="schema_version"):
        descriptor.load_descriptor(path)


def test_load_descriptor_wrong_schema_version(tmp_path):
    bad = {"schema_version": 99, "name": "FutureModel", "sources": []}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(bad))

    with pytest.raises(descriptor.DescriptorError, match="unsupported schema_version"):
        descriptor.load_descriptor(path)


def test_load_descriptor_file_not_found(tmp_path):
    with pytest.raises(descriptor.DescriptorError, match="not found"):
        descriptor.load_descriptor(tmp_path / "ghost.json")


# ---------------------------------------------------------------------------
# validate_descriptor edge cases
# ---------------------------------------------------------------------------


def test_validate_missing_name():
    data = {**VALID_DESCRIPTOR, "name": ""}
    issues = descriptor.validate_descriptor(data)
    assert any("name" in i for i in issues)


def test_validate_empty_sources():
    data = {**VALID_DESCRIPTOR, "sources": []}
    issues = descriptor.validate_descriptor(data)
    assert any("sources" in i for i in issues)


def test_validate_no_uuid_derivation():
    """No model_hash AND no url/params/size → validation error."""
    data = {
        "schema_version": 1,
        "name": "Incomplete",
        "sources": [{"source_type": "local_file", "url": ""}],
    }
    issues = descriptor.validate_descriptor(data)
    assert any("uuid" in i for i in issues)


# ---------------------------------------------------------------------------
# UUID derivation: hash vs provisional
# ---------------------------------------------------------------------------


def test_descriptor_uuid_from_hash():
    uuid = descriptor.descriptor_uuid(VALID_DESCRIPTOR)
    assert uuid == store.model_uuid("sha256:deadbeef1234")


def test_descriptor_uuid_provisional():
    uuid = descriptor.descriptor_uuid(PROVISIONAL_DESCRIPTOR)
    expected = store.provisional_model_uuid(
        "https://cdn.example.com/model.gguf",
        3.0,
        2_000_000_000,
    )
    assert uuid == expected


def test_save_provisional_descriptor():
    uuid = descriptor.save_descriptor(PROVISIONAL_DESCRIPTOR)
    assert uuid == descriptor.descriptor_uuid(PROVISIONAL_DESCRIPTOR)
    record = store.read_record("models", uuid)
    assert record["name"] == "ProvisionalModel-Q4_K_M"
