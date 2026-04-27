"""Fetch GGUFs from Hugging Face into the LM Studio layout.

Two modes:

    python -m bench.download [--config config.yaml] [--list]
        Read config, download every `gguf:` referenced under `models[]` and
        `judge` that is missing under `server.models_dir`.

    python -m bench.download <repo_id> <filename> [--config config.yaml] [--list]
        Ad-hoc. `repo_id` like `lmstudio-community/Qwen3.5-9B-GGUF`,
        `filename` like `Qwen3.5-9B-Q4_K_M.gguf` (may include subdirs).

Writes to `<models_dir>/<repo_id>/<filename>` so the existing `gguf:` paths
in `config.yaml` resolve without edits.

Auth: honors `HF_TOKEN` env or a cached `hf auth login` credential. Gated
repos (Llama, Gemma) surface a 401/403 with a pointer.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download

from bench.config import DEFAULT_CONFIG
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_models_dir(server_cfg: dict[str, Any]) -> Path:
    p = server_cfg.get("models_dir", "~/.lmstudio/models")
    return Path(os.path.expanduser(p)).resolve()


def split_gguf_path(gguf_rel: str) -> tuple[str, str]:
    """Split `<org>/<repo>/<file>...` into (`<org>/<repo>`, `<file>...`).

    Everything past the second slash is the in-repo filename (can be nested).
    """
    parts = gguf_rel.split("/", 2)
    if len(parts) < 3:
        raise ValueError(
            f"gguf path must be '<org>/<repo>/<file>.gguf' form: {gguf_rel!r}"
        )
    org, repo, filename = parts
    return f"{org}/{repo}", filename


def collect_from_config(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(repo_id, filename), ...] for every `gguf:` in models[] + judge.

    Preserves order; deduplicates (same repo+file appearing under both a model
    entry and the judge is common when reusing a model as judge).
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    entries: list[dict[str, Any]] = list(cfg.get("models") or [])
    judge = cfg.get("judge") or {}
    if judge.get("gguf"):
        entries.append(judge)
    for entry in entries:
        gguf = entry.get("gguf")
        if not gguf:
            continue
        pair = split_gguf_path(gguf)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def remote_size(api: HfApi, repo_id: str, filename: str) -> int | None:
    """Look up size in bytes for a single file. None if unknown."""
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
        return None
    for s in info.siblings or []:
        if s.rfilename == filename:
            return getattr(s, "size", None)
    return None


def human_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024
    return f"{n} B"


def target_path(models_dir: Path, repo_id: str, filename: str) -> Path:
    return models_dir / repo_id / filename


def download_one(
    models_dir: Path,
    repo_id: str,
    filename: str,
    token: str | None,
) -> Path:
    """Download or no-op if already present. Returns final path."""
    dest = target_path(models_dir, repo_id, filename)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    local_dir = models_dir / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)
    path_str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(path_str)


def _print_plan(
    pairs: list[tuple[str, str]],
    models_dir: Path,
    api: HfApi,
) -> int:
    """Print what would be fetched + totals. Returns count to fetch."""
    to_fetch = 0
    total_bytes: int | None = 0
    for repo_id, filename in pairs:
        dest = target_path(models_dir, repo_id, filename)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [ok] {repo_id}/{filename}  (present)")
            continue
        size = remote_size(api, repo_id, filename)
        if size is None:
            total_bytes = None
        elif total_bytes is not None:
            total_bytes += size
        print(f"  [fetch] {repo_id}/{filename}  ({human_bytes(size)})")
        to_fetch += 1
    if to_fetch == 0:
        print("nothing to fetch.")
    else:
        print(f"\n{to_fetch} file(s) to fetch, total {human_bytes(total_bytes)}.")
    return to_fetch


def _format_error(repo_id: str, filename: str, e: Exception) -> str:
    if isinstance(e, GatedRepoError):
        return (
            f"{repo_id}/{filename}: gated repo — run `hf auth login` or set "
            f"HF_TOKEN to a token with access."
        )
    if isinstance(e, RepositoryNotFoundError):
        return f"{repo_id}/{filename}: repo not found (or private without auth)."
    if isinstance(e, EntryNotFoundError):
        return f"{repo_id}/{filename}: file not found in repo."
    if isinstance(e, HfHubHTTPError):
        status = getattr(e.response, "status_code", "?")
        if status in (401, 403):
            return (
                f"{repo_id}/{filename}: auth required (HTTP {status}). "
                f"Run `hf auth login` or set HF_TOKEN."
            )
        return f"{repo_id}/{filename}: HTTP {status}: {e}"
    return f"{repo_id}/{filename}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="bench.download",
        description="Fetch GGUFs from Hugging Face into the LM Studio layout.",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"Path to {DEFAULT_CONFIG} (for the no-arg config mode).")
    ap.add_argument("--list", action="store_true",
                    help="Dry-run: print what would be fetched, no download.")
    ap.add_argument("repo_id", nargs="?",
                    help="Ad-hoc: HF repo like org/name.")
    ap.add_argument("filename", nargs="?",
                    help="Ad-hoc: file within the repo (may contain '/').")
    args = ap.parse_args()

    if (args.repo_id is None) != (args.filename is None):
        ap.error("repo_id and filename must be given together (or neither).")

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    if args.repo_id:
        # Ad-hoc mode: still need models_dir from config (or fall back to default).
        cfg_path = Path(args.config)
        server_cfg = load_config(cfg_path).get("server", {}) if cfg_path.exists() else {}
        models_dir = resolve_models_dir(server_cfg)
        pairs = [(args.repo_id, args.filename)]
    else:
        cfg = load_config(Path(args.config))
        server_cfg = cfg.get("server", {})
        models_dir = resolve_models_dir(server_cfg)
        pairs = collect_from_config(cfg)
        if not pairs:
            print("no gguf entries found in config.", file=sys.stderr)
            return 0

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"models_dir: {models_dir}")

    if args.list:
        _print_plan(pairs, models_dir, api)
        return 0

    failures = 0
    for repo_id, filename in pairs:
        dest = target_path(models_dir, repo_id, filename)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[ok] {repo_id}/{filename}  (present)")
            continue
        print(f"[fetch] {repo_id}/{filename}")
        try:
            path = download_one(models_dir, repo_id, filename, token)
            print(f"[done] {path}")
        except Exception as e:
            failures += 1
            print(f"[err] {_format_error(repo_id, filename, e)}", file=sys.stderr)
            if args.repo_id:
                return 1  # ad-hoc: fail fast
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
