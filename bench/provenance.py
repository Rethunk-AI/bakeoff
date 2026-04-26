"""Run provenance collection for bakeoff results.

Collects local context with bounded subprocess calls and best-effort
fallbacks. Missing tools produce null fields plus warnings, not hard
failures.

HuggingFace enrichment is opt-in via `run.hf_enrichment` in config.yaml
or the `--hf-enrichment` CLI flag:
  off           — no network calls (default; safe for offline environments)
  best-effort   — failures append to provenance warnings, run continues
  strict        — any HF lookup failure raises RuntimeError and aborts
"""
from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from bench.config import config_hash


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stripped stdout, or None on any error."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _git_info(repo_root: Path) -> dict[str, Any]:
    root = str(repo_root)
    sha = _run(["git", "-C", root, "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_out = _run(["git", "-C", root, "status", "--porcelain"])
    dirty = bool(dirty_out)
    return {"sha": sha, "branch": branch, "dirty": dirty}


def _podman_version() -> str | None:
    out = _run(["podman", "version", "--format", "{{.Version}}"])
    if out:
        return out
    # fallback: first word of `podman version` free-form output
    full = _run(["podman", "version"])
    if full:
        m = re.search(r"Version:\s+(\S+)", full)
        if m:
            return m.group(1)
    return None


def _llama_swap_version(binary_dir: Path | None) -> str | None:
    candidates: list[Path] = []
    if binary_dir:
        for p in binary_dir.glob("llama-swap*"):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    out = _run([str(candidates[0]), "--version"])
    if out:
        # typically "llama-swap version X.Y.Z" or just "X.Y.Z"
        m = re.search(r"(\d+\.\d+[\.\d]*)", out)
        if m:
            return m.group(1)
    return None


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in packages:
        try:
            from importlib.metadata import version as pkg_version
            out[pkg] = pkg_version(pkg)
        except Exception:
            out[pkg] = None
    return out


def collect(
    cfg: dict[str, Any],
    seed: int,
    repo_root: Path,
    binary_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the provenance dict to embed in the result payload."""
    warnings: list[str] = []

    git = _git_info(repo_root)
    if git["sha"] is None:
        warnings.append("git SHA unavailable")

    podman_ver = _podman_version()
    if podman_ver is None:
        warnings.append("podman version unavailable")

    llama_swap_ver = _llama_swap_version(binary_dir)
    if llama_swap_ver is None:
        warnings.append("llama-swap version unavailable")

    pkg_versions = _package_versions(["httpx", "pyyaml", "huggingface_hub"])

    server_cfg = cfg.get("server") or {}

    return {
        "git": git,
        "config_hash": config_hash(cfg),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkg_versions,
        "podman_version": podman_ver,
        "llama_swap_version": llama_swap_ver,
        "server_image": server_cfg.get("image"),
        "warnings": warnings,
    }


def _hf_model_info(repo_id: str) -> dict[str, Any]:
    """Fetch HuggingFace model info. Raises on any failure (caller handles mode)."""
    from huggingface_hub import model_info
    info = model_info(repo_id)
    return {
        "hf_sha": getattr(info, "sha", None),
        "hf_tags": list(getattr(info, "tags", None) or []),
        "hf_pipeline_tag": getattr(info, "pipeline_tag", None),
        "hf_private": getattr(info, "private", None),
    }


def _hf_enrich(
    repo_id: str,
    mode: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Return HF fields for one repo_id, or None. mode: off|best-effort|strict."""
    if mode == "off":
        return None
    try:
        return _hf_model_info(repo_id)
    except Exception as e:
        msg = f"HuggingFace lookup failed for {repo_id!r}: {e}"
        if mode == "strict":
            raise RuntimeError(msg) from e
        warnings.append(msg)
        return None


def enrich_model_metadata(
    metadata: list[dict[str, Any]],
    mode: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Add HuggingFace fields to each model metadata entry.

    mode: off (no-op), best-effort (failures → warnings), strict (failures raise).
    Entries without a repo_id are left unchanged.
    """
    if mode == "off":
        return metadata
    result = []
    for m in metadata:
        repo_id = m.get("repo_id")
        hf = _hf_enrich(repo_id, mode, warnings) if repo_id else None
        result.append({**m, **(hf or {})})
    return result


def _infer_quantization(filename: str) -> str | None:
    """Extract quantization tag from a GGUF filename (e.g. Q4_K_M, IQ3_XS)."""
    m = re.search(r"((?:IQ|BF|F)\d+[_A-Z\d]*|Q\d+[_A-Z\d]+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_model_metadata(
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build normalized model_metadata list from config entries."""
    server_cfg = cfg.get("server") or {}
    default_ctx = server_cfg.get("ctx")
    models = cfg.get("models") or []
    out: list[dict[str, Any]] = []
    for m in models:
        gguf = m.get("gguf") or ""
        parts = gguf.split("/")
        repo_id = "/".join(parts[:2]) if len(parts) >= 3 else None
        filename = parts[-1] if parts else None
        quant = _infer_quantization(filename or "") if filename else None
        out.append({
            "id": m.get("id"),
            "alias": m.get("alias"),
            "gguf": gguf,
            "repo_id": repo_id,
            "filename": filename,
            "quantization": quant,
            "ctx": m.get("ctx") or default_ctx,
            "n_cpu_moe": m.get("n_cpu_moe"),
        })
    return out
