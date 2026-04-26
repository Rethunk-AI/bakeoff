# Tasks

## P0

- [x] Define stable keys for model rows: model ID, prompt ID, task ID.
- [x] Define stable keys for pairwise and scored judge rows.
- [x] Add resume source loading with clear errors for missing or malformed result JSON.
- [x] Validate resume compatibility for config identity, seed, tasks, prompts, models, and judge mode.
- [x] Add pending-cell planning for missing and errored model rows.
- [x] Keep model execution grouped by model for resumed pending cells.
- [x] Add resume metadata to reused and fresh rows.
- [x] Emit resumed benchmarks as new result files.
- [x] Add tests proving normal run phase order.
- [x] Add tests proving resumed run phase order.

## P1

- [ ] Add CLI selectors for `--rerun-errors`, `--rerun-missing`, model IDs, prompt IDs, and task IDs.
- [ ] Add judge resume planning for pairwise mode.
- [ ] Add judge resume planning for scored mode.
- [ ] Document resume examples in `HUMANS.md`.

## P2

- [ ] Consider a `--resume-strict-config-hash` flag for users who want byte-for-byte config matching.
- [ ] Consider a summary of reused, rerun, and skipped cells in Markdown/HTML reports.
