"""Package, validate, sign, and submit bakeoff result bundles.

The local benchmark flow still writes gitignored `results/run-*.json` files.
This module adds an explicit publication path: turn one result JSON into a
reviewable bundle, validate that bundle, optionally attach a Sigstore
signature, and open a publication PR against `Rethunk-AI/bakeoff-results`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "bakeoff-results/v1"
DEFAULT_RESULTS_REPO = "Rethunk-AI/bakeoff-results"
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class PublishError(ValueError):
    """Raised for validation and publication errors."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        raise PublishError(f"not found: {path}") from None
    except json.JSONDecodeError as e:
        raise PublishError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise PublishError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_run_id(payload: dict[str, Any]) -> str:
    raw = str(payload.get("run_id") or payload.get("timestamp") or "run")
    return SAFE_ID_RE.sub("-", raw).strip("-") or "run"


def _require_mapping(data: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}: required object")
        return {}
    return value


def _require_list(data: dict[str, Any], key: str, errors: list[str]) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{key}: required non-empty list")
        return []
    return value


def validate_result_payload(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for a public result payload."""
    errors: list[str] = []
    for key in ("run_id", "timestamp"):
        if not payload.get(key):
            errors.append(f"{key}: required")

    cfg = _require_mapping(payload, "config", errors)
    prov = _require_mapping(payload, "provenance", errors)
    models = _require_list(cfg, "models", errors) if cfg else []
    prompts = _require_list(cfg, "prompts", errors) if cfg else []
    tasks = _require_list(payload, "tasks", errors)
    records = _require_list(payload, "records", errors)
    metadata = _require_list(payload, "model_metadata", errors)

    if prov:
        if not prov.get("config_hash"):
            errors.append("provenance.config_hash: required")
        if prov.get("seed") is None:
            errors.append("provenance.seed: required")
        git = prov.get("git")
        if not isinstance(git, dict) or not git.get("sha"):
            errors.append("provenance.git.sha: required")

    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"config.models[{i}]: required object")
            continue
        if not model.get("id"):
            errors.append(f"config.models[{i}].id: required")
        if not model.get("gguf"):
            errors.append(f"config.models[{i}].gguf: required")

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict) or not prompt.get("id"):
            errors.append(f"config.prompts[{i}].id: required")

    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or not task.get("id"):
            errors.append(f"tasks[{i}].id: required")

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"records[{i}]: required object")
            continue
        for key in ("task_id", "prompt_id", "model_id"):
            if not record.get(key):
                errors.append(f"records[{i}].{key}: required")

    for i, item in enumerate(metadata):
        if not isinstance(item, dict):
            errors.append(f"model_metadata[{i}]: required object")
            continue
        if not item.get("id"):
            errors.append(f"model_metadata[{i}].id: required")
        if not item.get("gguf"):
            errors.append(f"model_metadata[{i}].gguf: required")

    return errors


def _manifest_file_errors(bundle_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return ["manifest.files: required object"]
    for rel, meta in files.items():
        if "/" in rel and any(part in {"", ".", ".."} for part in Path(rel).parts):
            errors.append(f"manifest.files.{rel}: unsafe path")
            continue
        if not isinstance(meta, dict) or not meta.get("sha256"):
            errors.append(f"manifest.files.{rel}.sha256: required")
            continue
        path = bundle_dir / rel
        if not path.is_file():
            errors.append(f"{rel}: missing")
            continue
        actual = _sha256_file(path)
        if actual != meta["sha256"]:
            errors.append(f"{rel}: sha256 mismatch")
    return errors


def validate_bundle(bundle_dir: Path) -> list[str]:
    """Return validation errors for a packaged result bundle directory."""
    errors: list[str] = []
    manifest_path = bundle_dir / "manifest.json"
    result_path = bundle_dir / "result.json"
    if not manifest_path.is_file():
        return [f"{manifest_path}: missing"]
    if not result_path.is_file():
        return [f"{result_path}: missing"]

    try:
        manifest = _load_json(manifest_path)
        payload = _load_json(result_path)
    except PublishError as e:
        return [str(e)]

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest.schema_version: expected {SCHEMA_VERSION!r}")
    if not manifest.get("run_id"):
        errors.append("manifest.run_id: required")
    if not manifest.get("created_at"):
        errors.append("manifest.created_at: required")

    errors.extend(validate_result_payload(payload))
    errors.extend(_manifest_file_errors(bundle_dir, manifest))

    signature = manifest.get("signature")
    if signature is not None:
        if not isinstance(signature, dict):
            errors.append("manifest.signature: required object when present")
        elif signature.get("required") and not signature.get("path"):
            errors.append("manifest.signature.path: required when signature.required=true")
    return errors


def validate_path(path: Path) -> list[str]:
    if path.is_dir():
        return validate_bundle(path)
    return validate_result_payload(_load_json(path))


def _emit_reports(payload: dict[str, Any], bundle_dir: Path) -> None:
    from bench.report import emit_html, emit_markdown

    emit_markdown(payload, bundle_dir / "summary.md")
    emit_html(payload, bundle_dir / "dashboard.html")


def _build_manifest(bundle_dir: Path, payload: dict[str, Any], signed: bool) -> dict[str, Any]:
    files = {}
    for rel in ("result.json", "summary.md", "dashboard.html"):
        path = bundle_dir / rel
        if path.is_file():
            files[rel] = {"sha256": _sha256_file(path)}
    sig_path = bundle_dir / "signature.sigstore.json"
    if sig_path.is_file():
        files["signature.sigstore.json"] = {"sha256": _sha256_file(sig_path)}

    prov = payload.get("provenance") or {}
    cfg = payload.get("config") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "run_id": payload.get("run_id"),
        "timestamp": payload.get("timestamp"),
        "config_hash": prov.get("config_hash"),
        "judge_mode": (cfg.get("judge") or {}).get("mode"),
        "model_ids": [m.get("id") for m in (cfg.get("models") or []) if isinstance(m, dict)],
        "files": files,
        "signature": {
            "kind": "sigstore-bundle",
            "path": "signature.sigstore.json" if sig_path.is_file() else None,
            "signed_file": "result.json",
            "required": signed,
        },
    }


def _sign_result(bundle_dir: Path, cosign: str) -> None:
    result = bundle_dir / "result.json"
    bundle = bundle_dir / "signature.sigstore.json"
    cmd = [cosign, "sign-blob", str(result), "--bundle", str(bundle), "--yes"]
    subprocess.run(cmd, check=True)


def package_result(
    result_path: Path,
    output_dir: Path | None = None,
    *,
    force: bool = False,
    sign: bool = False,
    cosign: str = "cosign",
) -> Path:
    payload = _load_json(result_path)
    errors = validate_result_payload(payload)
    if errors:
        raise PublishError("invalid result:\n- " + "\n- ".join(errors))

    bundle_dir = output_dir or result_path.parent / "publish" / _safe_run_id(payload)
    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not force:
        raise PublishError(f"{bundle_dir} exists; pass --force to overwrite")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(bundle_dir / "result.json", payload)
    _emit_reports(payload, bundle_dir)
    _write_json(bundle_dir / "manifest.json", _build_manifest(bundle_dir, payload, signed=sign))
    if sign:
        _sign_result(bundle_dir, cosign)
        _write_json(bundle_dir / "manifest.json", _build_manifest(bundle_dir, payload, signed=True))

    errors = validate_bundle(bundle_dir)
    if errors:
        raise PublishError("invalid bundle:\n- " + "\n- ".join(errors))
    return bundle_dir


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def submit_bundle(
    bundle_dir: Path,
    *,
    checkout: Path,
    repo: str = DEFAULT_RESULTS_REPO,
    branch: str | None = None,
    dry_run: bool = False,
) -> Path:
    errors = validate_bundle(bundle_dir)
    if errors:
        raise PublishError("invalid bundle:\n- " + "\n- ".join(errors))

    manifest = _load_json(bundle_dir / "manifest.json")
    run_id = str(manifest["run_id"])
    safe_id = SAFE_ID_RE.sub("-", run_id).strip("-") or "run"
    dest = checkout / "submissions" / safe_id

    if dry_run:
        print(f"[publish] would submit {bundle_dir} -> {repo}:{dest.relative_to(checkout)}")
        return dest

    if not checkout.exists():
        _run(["gh", "repo", "clone", repo, str(checkout)])
    if not (checkout / ".git").exists():
        raise PublishError(f"{checkout} is not a git checkout")

    branch_name = branch or f"submit/{safe_id}"
    _run(["git", "fetch", "origin"], cwd=checkout)
    _run(["git", "checkout", "-B", branch_name, "origin/main"], cwd=checkout)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_dir, dest)

    rel_dest = dest.relative_to(checkout)
    _run(["git", "add", str(rel_dest)], cwd=checkout)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not status:
        print("[publish] no changes to submit")
        return dest
    _run([
        "git",
        "-c", "user.name=Bastion Agent",
        "-c", "user.email=bastion-agent@rethunk.tech",
        "commit",
        "-m", f"results: submit {safe_id}",
    ], cwd=checkout)
    _run(["git", "push", "-u", "origin", branch_name], cwd=checkout)
    _run([
        "gh", "pr", "create",
        "--repo", repo,
        "--title", f"results: submit {safe_id}",
        "--body", f"Submit signed bakeoff result bundle `{safe_id}`.",
    ], cwd=checkout)
    return dest


def _print_errors(errors: list[str]) -> int:
    for error in errors:
        print(f"[publish] error: {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate, package, sign, and submit bakeoff results.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate a result JSON file or packaged bundle directory")
    v.add_argument("path", type=Path)

    p = sub.add_parser("package", help="Create a publication bundle from a result JSON file")
    p.add_argument("result", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--sign", action="store_true", help="Sign result.json with cosign sign-blob")
    p.add_argument("--cosign", default="cosign")

    s = sub.add_parser("submit", help="Submit a packaged bundle to the results repository")
    s.add_argument("bundle", type=Path)
    s.add_argument("--checkout", type=Path, default=Path("../bakeoff-results"))
    s.add_argument("--repo", default=DEFAULT_RESULTS_REPO)
    s.add_argument("--branch", default=None)
    s.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "validate":
            errors = validate_path(args.path)
            if errors:
                return _print_errors(errors)
            print(f"[publish] valid: {args.path}")
            return 0
        if args.cmd == "package":
            out = package_result(
                args.result,
                args.output_dir,
                force=args.force,
                sign=args.sign,
                cosign=args.cosign,
            )
            print(f"[publish] bundle: {out}")
            return 0
        if args.cmd == "submit":
            dest = submit_bundle(
                args.bundle,
                checkout=args.checkout,
                repo=args.repo,
                branch=args.branch,
                dry_run=args.dry_run,
            )
            print(f"[publish] submitted: {dest}")
            return 0
    except (PublishError, subprocess.CalledProcessError) as e:
        print(f"[publish] error: {e}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
