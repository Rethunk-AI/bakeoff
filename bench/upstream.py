"""Upstream model-metadata pull: submitted URL -> validated descriptor dict.

Only Hugging Face is supported; any other host is rejected with an error rather
than guessed at. Network access is confined to :func:`fetch_json`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from bench import store

# Public model-info endpoint (https://huggingface.co/docs/hub/api, "GET /api/models/{repo_id}").
# ``blobs=true`` adds per-file ``size`` (and ``lfs.sha256``) to ``siblings``; the response
# also carries ``gguf`` {total, architecture, context_length}, ``cardData.license`` and
# ``tags`` including ``license:<id>``. Shape verified against the live API; the recorded
# copy is tests/fixtures/hf_qwen3_8b_gguf.json.
HF_API = "https://huggingface.co/api/models"
HF_HOSTS = {"huggingface.co", "www.huggingface.co"}
TIMEOUT = httpx.Timeout(15.0, connect=5.0)

_QUANT = re.compile(r"(?:^|[-_.])(i?q\d(?:_[a-z0-9]+)*|bf16|f16|f32)(?=[-_.]|$)", re.IGNORECASE)
_QUANT_ALIAS = {"f16": "fp16", "f32": "fp32"}


class UpstreamError(ValueError):
    """Raised when a submitted URL cannot be resolved to valid upstream metadata."""


@dataclass(frozen=True)
class HFRef:
    repo: str
    filename: str | None


def parse_hf_url(url: str) -> HFRef:
    """Parse ``https://huggingface.co/<org>/<repo>[/(blob|resolve)/<rev>/<file>]``."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in HF_HOSTS:
        raise UpstreamError(f"unsupported URL {url!r}: only https://huggingface.co is supported")
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise UpstreamError(f"{url!r}: expected https://huggingface.co/<org>/<repo>")
    repo = f"{parts[0]}/{parts[1]}"
    if len(parts) == 2 or parts[2] == "tree":
        return HFRef(repo, None)
    if parts[2] in ("blob", "resolve") and len(parts) >= 5:
        return HFRef(repo, "/".join(parts[4:]))
    raise UpstreamError(f"{url!r}: unrecognised Hugging Face URL shape")


def fetch_json(url: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """GET *url* and return the JSON object; the only network call in this module."""
    try:
        if client is None:
            with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
                resp = c.get(url)
        else:
            resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise UpstreamError(f"upstream fetch failed for {url}: {e}") from e
    if not isinstance(data, dict):
        raise UpstreamError(f"upstream {url}: expected a JSON object")
    return data


def _license(info: dict[str, Any]) -> str | None:
    card = info.get("cardData")
    lic = card.get("license") if isinstance(card, dict) else None
    if isinstance(lic, str) and lic:
        return lic
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.removeprefix("license:")
    return None


def _pick_file(info: dict[str, Any], filename: str | None) -> dict[str, Any] | None:
    siblings = [s for s in info.get("siblings") or [] if isinstance(s, dict)]
    if filename is not None:
        for s in siblings:
            if s.get("rfilename") == filename:
                return s
        raise UpstreamError(f"file {filename!r} not found in {info.get('id')}")
    ggufs = [s for s in siblings if str(s.get("rfilename", "")).lower().endswith(".gguf")]
    if len(ggufs) > 1:
        names = ", ".join(sorted(str(s["rfilename"]) for s in ggufs))
        raise UpstreamError(f"{info.get('id')} has several GGUF files; submit one: {names}")
    return ggufs[0] if ggufs else None


def build_descriptor(info: dict[str, Any], ref: HFRef, url: str) -> dict[str, Any]:
    """Map an HF model-info response to a descriptor dict (not yet validated)."""
    repo = ref.repo
    author = str(info.get("author") or repo.split("/")[0])
    file = _pick_file(info, ref.filename)
    fname = str(file["rfilename"]) if file else None
    is_gguf = bool(fname and fname.lower().endswith(".gguf"))
    gguf = info.get("gguf") if isinstance(info.get("gguf"), dict) else {}

    fmt = (
        "GGUF"
        if is_gguf
        else ("SafeTensors" if "safetensors" in (info.get("tags") or []) else None)
    )
    quant = None
    if fname:
        m = _QUANT.search(fname.rsplit("/", 1)[-1])
        if m:
            q = m.group(1).lower()
            quant = _QUANT_ALIAS.get(q, q)
    total = gguf.get("total")
    lfs = file.get("lfs") if file and isinstance(file.get("lfs"), dict) else {}
    sha = lfs.get("sha256")
    stem = (
        fname.rsplit("/", 1)[-1].removesuffix(".gguf") if is_gguf and fname else repo.split("/")[1]
    )

    return {
        "schema_version": store.SCHEMA_VERSION,
        "name": stem,
        "creator": {
            "name": author.lower(),
            "display_name": author,
            "homepage": f"https://huggingface.co/{author}",
            "service_identifiers": {"huggingface": author},
            "provisional": False,
        },
        "model_hash": f"sha256:{sha}" if sha else None,
        "parameter_count_b": round(total / 1e9, 3) if isinstance(total, int) else None,
        "active_parameter_count_b": None,
        "architecture": gguf.get("architecture"),
        "file_format": fmt,
        "quantization": quant,
        "context_length_default": None,
        "context_length_min": None,
        "context_length_max": gguf.get("context_length"),
        "model_source_mtime": None,
        "model_source_size": file.get("size") if file else None,
        "release_date": None,
        "version": None,
        "description": None,
        "predecessor_model_id": None,
        "provisional": not sha,
        "sources": [
            {
                "source_type": "huggingface",
                "url": url,
                "source_metadata": {
                    "hf_repo": repo,
                    "hf_commit": info.get("sha"),
                    "hf_file": fname,
                    "downloads": info.get("downloads"),
                    "license": _license(info),
                },
            }
        ],
    }


def validate_upstream(info: dict[str, Any], descriptor: dict[str, Any]) -> list[str]:
    """Checks that only make sense on freshly pulled upstream data."""
    issues: list[str] = []
    if info.get("private") or info.get("disabled"):
        issues.append("repo: private or disabled upstream")
    if info.get("gated"):
        issues.append("repo: gated upstream (needs an access grant to download)")
    if descriptor["file_format"] is None:
        issues.append("file_format: not GGUF or SafeTensors")
    meta = descriptor["sources"][0]["source_metadata"]
    if not meta["license"]:
        issues.append("license: upstream declares none")
    size = descriptor["model_source_size"]
    if size is None and descriptor["file_format"] == "GGUF":
        issues.append("model_source_size: no file size reported")
    elif size is not None and (not isinstance(size, int) or size <= 0):
        issues.append(f"model_source_size: invalid {size!r}")
    return issues


def pull_descriptor(url: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetch upstream metadata for *url* and return a descriptor passing every check."""
    ref = parse_hf_url(url)
    info = fetch_json(f"{HF_API}/{ref.repo}?blobs=true", client)
    desc = build_descriptor(info, ref, url)
    issues = validate_upstream(info, desc)
    if issues:
        raise UpstreamError("invalid upstream metadata:\n- " + "\n- ".join(issues))
    return desc
