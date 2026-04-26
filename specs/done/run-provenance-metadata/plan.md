# Plan

## Implementation Slices

- Add `bench.provenance` with helpers for git, config hash, Python/package versions, platform, GPU command availability, podman, llama-swap, and container image.
- Add model metadata normalization from config and server defaults.
- Add optional HuggingFace enrichment behind a config and/or CLI switch with deterministic timeout and mockable API boundary.
- Extend the result payload with `provenance` and `model_metadata`.
- Update Markdown and HTML report headers to render provenance when present and skip gracefully for legacy payloads.
- Add tests for local provenance, model metadata inference, HuggingFace enrichment success/failure, and legacy report compatibility.
- Document provenance fields and enrichment behavior in `HUMANS.md`.

## Test Plan

- `uv run pytest tests/test_provenance.py tests/test_report.py`
- `uv run python -m bench.runner --config config.yaml --dry-run`
- `uv run mypy bench/`
- `uv run pre-commit run --all-files`
