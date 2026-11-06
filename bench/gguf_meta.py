"""Read-only, bounded GGUF (v2/v3) header metadata reader.

Only the key/value section is parsed; tensor data is never read. Every length
read from the file is capped so a corrupt or hostile header cannot make the
reader allocate or loop without bound.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

MAX_KV_COUNT = 100_000
MAX_STRING_BYTES = 1 << 20
MAX_ARRAY_LEN = 10_000_000

# Keys kept in the result; everything else is skipped without being stored.
_WANTED_PREFIXES = ("general.",)
_WANTED_SUFFIXES = (".context_length",)

# gguf value type id -> struct format (little-endian); 8 = string, 9 = array.
_SCALAR = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_T_STRING = 8
_T_ARRAY = 9

# llama.cpp LLAMA_FTYPE_* ids, lowercased to match descriptor `quantization`.
FILE_TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
    19: "iq2_xxs",
    20: "iq2_xs",
    21: "q2_k_s",
    22: "iq3_xs",
    23: "iq3_xxs",
    24: "iq1_s",
    25: "iq4_nl",
    26: "iq3_s",
    27: "iq3_m",
    28: "iq2_s",
    29: "iq2_m",
    30: "iq4_xs",
    31: "iq1_m",
    32: "bf16",
}


class GGUFError(ValueError):
    """Raised when a file is not a readable GGUF header."""


def _read(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFError("truncated GGUF header")
    return buf


def _unpack(f: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]


def _string(f: BinaryIO) -> str:
    n = _unpack(f, "<Q")
    if n > MAX_STRING_BYTES:
        raise GGUFError(f"string length {n} exceeds {MAX_STRING_BYTES}")
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, vtype: int, keep: bool) -> Any:
    if vtype in _SCALAR:
        return _unpack(f, _SCALAR[vtype])
    if vtype == _T_STRING:
        return _string(f)
    if vtype == _T_ARRAY:
        etype = _unpack(f, "<I")
        n = _unpack(f, "<Q")
        if n > MAX_ARRAY_LEN:
            raise GGUFError(f"array length {n} exceeds {MAX_ARRAY_LEN}")
        if etype in _SCALAR and not keep:
            f.seek(n * struct.calcsize(_SCALAR[etype]), 1)
            return None
        items = [_value(f, etype, keep) for _ in range(n)]
        return items if keep else None
    raise GGUFError(f"unknown value type {vtype}")


def read_metadata(path: Path) -> dict[str, Any]:
    """Return the ``general.*`` and ``*.context_length`` KV pairs of a GGUF file."""
    try:
        f = path.open("rb")
    except OSError as e:
        raise GGUFError(f"cannot open {path}: {e.strerror}") from e
    with f:
        if _read(f, 4) != MAGIC:
            raise GGUFError(f"{path}: not a GGUF file (bad magic)")
        version = _unpack(f, "<I")
        if version not in SUPPORTED_VERSIONS:
            raise GGUFError(f"{path}: unsupported GGUF version {version}")
        _unpack(f, "<Q")  # tensor_count
        kv_count = _unpack(f, "<Q")
        if kv_count > MAX_KV_COUNT:
            raise GGUFError(f"{path}: kv count {kv_count} exceeds {MAX_KV_COUNT}")
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(f)
            vtype = _unpack(f, "<I")
            keep = key.startswith(_WANTED_PREFIXES) or key.endswith(_WANTED_SUFFIXES)
            value = _value(f, vtype, keep)
            if keep:
                meta[key] = value
        return meta


def cross_check(descriptor: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    """Compare descriptor fields against GGUF metadata; one issue per mismatch.

    A field is compared only when both sides declare it.
    """
    issues: list[str] = []

    ftype = meta.get("general.file_type")
    declared_q = descriptor.get("quantization")
    if ftype is not None and declared_q:
        embedded_q = FILE_TYPES.get(ftype, f"unknown({ftype})")
        if str(declared_q).lower() != embedded_q:
            issues.append(f"quantization: descriptor {declared_q!r} != gguf {embedded_q!r}")

    arch = meta.get("general.architecture")
    ctx = meta.get(f"{arch}.context_length") if isinstance(arch, str) else None
    declared_ctx = descriptor.get("context_length_max")
    if ctx is not None and declared_ctx is not None and declared_ctx != ctx:
        issues.append(f"context_length_max: descriptor {declared_ctx} != gguf {ctx}")

    count = meta.get("general.parameter_count")
    declared_p = descriptor.get("parameter_count_b")
    if count is not None and declared_p is not None:
        embedded_b = count / 1e9
        # Published sizes are rounded ("8B" for 8.19B), so allow 5%.
        if abs(embedded_b - declared_p) > 0.05 * embedded_b:
            issues.append(f"parameter_count_b: descriptor {declared_p} != gguf {embedded_b:.2f}")
    return issues
