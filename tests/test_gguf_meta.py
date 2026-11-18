"""GGUF header parser: keep/skip rules, bounds, and descriptor cross-check."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from bench.gguf_meta import (
    MAX_ARRAY_LEN,
    MAX_KV_COUNT,
    MAX_STRING_BYTES,
    GGUFError,
    cross_check,
    read_metadata,
)


def _key(name: str) -> bytes:
    raw = name.encode()
    return struct.pack("<Q", len(raw)) + raw


def _str_payload(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key: str, value: str) -> bytes:
    return _key(key) + struct.pack("<I", 8) + _str_payload(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _key(key) + struct.pack("<II", 4, value)


def _kv_u64(key: str, value: int) -> bytes:
    return _key(key) + struct.pack("<IQ", 10, value)


def _header(kvs: list[bytes], *, version: int = 3, tensor_count: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, tensor_count, len(kvs)) + b"".join(kvs)


def _write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    return path


def test_reader_keeps_general_and_context_keys_and_skips_the_rest(tmp_path):
    # Unwanted scalar arrays are seek-skipped; unwanted string arrays are parsed then dropped.
    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_u32("general.file_type", 15),
        _kv_u64("general.parameter_count", 8_190_000_000),
        _kv_u32("llama.context_length", 32768),
        _key("general.tags")
        + struct.pack("<IIQ", 9, 8, 2)
        + _str_payload("chat")
        + _str_payload("base"),
        _key("tokenizer.pre") + struct.pack("<IIQ", 9, 0, 3) + b"\x01\x02\x03",
        _key("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 1) + _str_payload("hello"),
    ]
    meta = read_metadata(_write(tmp_path / "m.gguf", _header(kvs)))
    assert meta["general.architecture"] == "llama"
    assert meta["general.file_type"] == 15
    assert meta["general.parameter_count"] == 8_190_000_000
    assert meta["llama.context_length"] == 32768
    assert meta["general.tags"] == ["chat", "base"]
    assert "tokenizer.pre" not in meta
    assert "tokenizer.ggml.tokens" not in meta

    matching = {
        "quantization": "q4_k_m",
        "context_length_max": 32768,
        "parameter_count_b": 8.19,
    }
    assert cross_check(matching, meta) == []
    drifted = {**matching, "parameter_count_b": 1.0}
    issues = cross_check(drifted, meta)
    assert any("parameter_count_b" in issue for issue in issues)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"GGUF", "truncated"),
        (b"GGUF" + struct.pack("<IQQ", 1, 0, 0), "unsupported GGUF version"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, MAX_KV_COUNT + 1), "kv count"),
        (struct.pack("<Q", MAX_STRING_BYTES + 1), "string length"),
        (
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)
            + _key("general.x")
            + struct.pack("<IIQ", 9, 0, MAX_ARRAY_LEN + 1),
            "array length",
        ),
        (
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _key("general.x") + struct.pack("<I", 99),
            "unknown value type",
        ),
    ],
)
def test_corrupt_or_hostile_headers_are_rejected(tmp_path, raw, match):
    if raw[:4] != b"GGUF":
        raw = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + raw
    path = _write(tmp_path / "bad.gguf", raw)
    with pytest.raises(GGUFError, match=match):
        read_metadata(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(GGUFError, match="cannot open"):
        read_metadata(tmp_path / "no-such.gguf")
