# Plan

## Implementation Slices

- Add stable matrix key helpers for model records and judge records.
- Add result-loading and compatibility checks for resume sources.
- Add runner CLI flags for resume source and rerun policy.
- Build pending model cells from current tasks/prompts/models minus reusable prior rows.
- Execute pending cells per model, preserving warmup and phase ordering.
- Merge reused and fresh records into a new payload with resume metadata.
- Add judge resume planning for pairwise and scored modes.
- Add tests around normal and resumed phase order with fake clients.
- Document resume workflows and limitations in `HUMANS.md`.

## Test Plan

- `uv run pytest tests/test_runner_resume.py tests/test_runner_phase_order.py`
- `uv run python -m bench.runner --config config.yaml --dry-run`
- `uv run mypy bench/`
- `uv run pre-commit run --all-files`
