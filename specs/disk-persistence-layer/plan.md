# Plan

## Implementation Slices

1. `bench/store.py` — shared atomic I/O, audit stamping, UUID5 helpers.
2. `bench/descriptor.py` — model descriptor reader/validator/persister (seed JSON only).
3. Tests — `tests/test_store.py`, `tests/test_descriptor.py`.
4. Spec — `specs/disk-persistence-layer/{spec,plan,tasks}.md` + `specs/index.md` row.
5. Docs — AGENTS.md (design invariants + layout), HUMANS.md (`BAKEOFF_DATA_DIR` env var).

## Test Plan

- `uv run ruff check bench/store.py bench/descriptor.py tests/test_store.py tests/test_descriptor.py`
- `uv run pyrefly check bench/store.py bench/descriptor.py`
- `uv run pytest tests/test_store.py tests/test_descriptor.py -q`
- `uv run pytest -q` (full suite; no new regressions)
