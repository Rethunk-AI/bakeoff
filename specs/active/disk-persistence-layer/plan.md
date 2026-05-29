# Plan

## Implementation Slices

1. `bench/store.py` — shared atomic I/O, audit stamping, UUID5 helpers.
2. `bench/descriptor.py` — model descriptor reader/validator/persister (seed JSON only).
3. `bench/queue.py` — disk-backed opt-in run queue with race-safe claim.
4. Tests — `tests/test_store.py`, `tests/test_descriptor.py`, `tests/test_queue.py`.
5. Spec — `specs/active/disk-persistence-layer/{spec,plan,tasks}.md` + `specs/index.md` row.
6. Docs — AGENTS.md (design invariants + layout), HUMANS.md (`BAKEOFF_DATA_DIR` env var).

## Test Plan

- `uv run ruff check bench/store.py bench/descriptor.py bench/queue.py tests/test_store.py tests/test_descriptor.py tests/test_queue.py`
- `uv run mypy bench/store.py bench/descriptor.py bench/queue.py`
- `uv run pytest tests/test_store.py tests/test_descriptor.py tests/test_queue.py -q`
- `uv run pytest -q` (full suite; no new regressions)
