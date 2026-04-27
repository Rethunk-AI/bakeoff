"""Config loading and validation for bakeoff runs.

Provides a single validate-before-run gate so invalid configs fail early
with path-oriented error messages rather than at proxy startup or mid-matrix.
Errors are aggregated so operators can fix multiple issues in one edit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_VALID_JUDGE_MODES = {"pairwise_all", "scored"}
_VALID_DOMAINS = {"qa", "code", "summarize", "classify"}
DEFAULT_CONFIG = "config.yaml"


class ConfigError(ValueError):
    """Raised when the config cannot be loaded (parse failure, missing file)."""


@dataclass
class ValidationIssue:
    """A single config validation finding with a dotted path and message."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def load_config(path: Path) -> dict[str, Any]:
    """Load and YAML-parse config. Raises ConfigError on failure."""
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return data


def config_hash(cfg: dict[str, Any]) -> str:
    """16-hex SHA256 fingerprint of canonical JSON config representation."""
    canonical = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def resolve_models_dir(server_cfg: dict[str, Any]) -> Path:
    """Resolve models directory path from server config, expanding ~ and making absolute."""
    import os
    p = server_cfg.get("models_dir", "~/.lmstudio/models")
    return Path(os.path.expanduser(p)).resolve()


def _check_id(item: dict[str, Any], item_idx: int, field_name: str, path_prefix: str,
              seen_ids: set[str], issues: list[ValidationIssue]) -> None:
    """Check ID presence and uniqueness; report issues."""
    item_id = item.get("id")
    if not item_id:
        issues.append(ValidationIssue(f"{path_prefix}[{item_idx}].id", "required"))
    elif item_id in seen_ids:
        issues.append(ValidationIssue(f"{path_prefix}[{item_idx}].id", f"duplicate {field_name} id {item_id!r}"))
    else:
        seen_ids.add(str(item_id))


def _check_gguf_shape(gguf: str, path: str, issues: list[ValidationIssue]) -> None:
    basename = gguf.rsplit("/", 1)[-1]
    if basename.lower().startswith("mmproj"):
        issues.append(ValidationIssue(
            path,
            f"{basename!r} is an mmproj vision projector, not a standalone text model",
        ))
        return
    parts = gguf.split("/")
    if len(parts) < 3 or not parts[-1].endswith(".gguf"):
        issues.append(ValidationIssue(
            path,
            f"expected '<org>/<repo>/<file>.gguf' form, got {gguf!r}",
        ))


def validate_config(cfg: dict[str, Any]) -> list[ValidationIssue]:
    """Validate config and return a list of issues. Empty list means valid.

    Covers: required sections, duplicate IDs, GGUF path shape, mmproj
    rejection, judge mode, numeric fields, non-empty models/prompts/domains.
    Does NOT check that GGUF files exist on disk — that belongs to the
    provenance and metadata spec.
    """
    issues: list[ValidationIssue] = []

    def err(path: str, msg: str) -> None:
        issues.append(ValidationIssue(path, msg))

    # Required sections
    for section in ("server", "models", "prompts", "dataset"):
        if not cfg.get(section):
            err(section, "required section missing or empty")

    # Dataset
    ds = cfg.get("dataset") or {}
    n = ds.get("n")
    if n is None:
        err("dataset.n", "required")
    elif not isinstance(n, int) or n <= 0:
        err("dataset.n", f"must be a positive integer, got {n!r}")

    domains = ds.get("domains") or []
    if not domains:
        err("dataset.domains", "must be a non-empty list")
    else:
        for d in domains:
            if d not in _VALID_DOMAINS:
                err("dataset.domains", f"unknown domain {d!r}; valid: {sorted(_VALID_DOMAINS)}")

    # Models
    models = cfg.get("models") or []
    seen_model_ids: set[str] = set()
    for i, m in enumerate(models):
        _check_id(m, i, "model", "models", seen_model_ids, issues)
        prefix = f"models[{i}]"

        gguf = m.get("gguf")
        if not gguf:
            err(f"{prefix}.gguf", "required")
        else:
            _check_gguf_shape(str(gguf), f"{prefix}.gguf", issues)

    # Prompts
    prompts = cfg.get("prompts") or []
    seen_pids: set[str] = set()
    for i, p in enumerate(prompts):
        _check_id(p, i, "prompt", "prompts", seen_pids, issues)

    # Judge
    judge = cfg.get("judge") or {}
    if judge.get("enabled"):
        mode = judge.get("mode", "pairwise_all")
        if mode not in _VALID_JUDGE_MODES:
            err("judge.mode", f"unknown mode {mode!r}; valid: {sorted(_VALID_JUDGE_MODES)}")

        judge_gguf = judge.get("gguf")
        if judge_gguf:
            _check_gguf_shape(str(judge_gguf), "judge.gguf", issues)

        judge_id = str(judge.get("id") or "judge")
        if judge_id in seen_model_ids:
            err("judge.id",
                f"judge id {judge_id!r} collides with a model id; set judge.id to a distinct value")

    # Server — positive numeric fields
    server = cfg.get("server") or {}
    for field in ("ctx", "ngl", "ubatch", "boot_timeout_s", "swap_port", "backend_start_port"):
        v = server.get(field)
        if v is not None and (not isinstance(v, (int, float)) or v <= 0):
            err(f"server.{field}", f"must be a positive number, got {v!r}")

    # Cost
    cost = cfg.get("cost") or {}
    if cost.get("enabled"):
        rate = cost.get("kwh_rate_usd")
        if rate is not None and (not isinstance(rate, (int, float)) or rate < 0):
            err("cost.kwh_rate_usd", f"must be a non-negative number, got {rate!r}")
        hz = cost.get("sample_hz")
        if hz is not None and (not isinstance(hz, (int, float)) or hz <= 0):
            err("cost.sample_hz", f"must be a positive number, got {hz!r}")

    # Output
    out_cfg = cfg.get("output") or {}
    out_dir = out_cfg.get("dir")
    if out_dir is not None and not isinstance(out_dir, str):
        err("output.dir", f"must be a string, got {out_dir!r}")

    return issues
