# Plan

## Implementation Slices

- Extract or expose report rollup helpers needed by comparison.
- Add result JSON loading and compatibility warning helpers.
- Implement two-run comparison data model with per-model metric deltas.
- Add Markdown comparison renderer.
- Add CLI entry point for comparison output.
- Add tests for numeric deltas, missing metrics, judge rollups, and warnings.
- Document comparison examples in `HUMANS.md`.

## Test Plan

- `uv run pytest tests/test_compare.py tests/test_report.py`
- `uv run python -m bench.compare results/a.json results/b.json --output /tmp/compare.md` with fixtures or generated samples.
- `uv run mypy bench/`
- `uv run pre-commit run --all-files`
