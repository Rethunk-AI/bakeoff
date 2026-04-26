# Plan

## Implementation Slices

- Add a `Recommended benchmark profiles` section to `HUMANS.md`.
- Keep README unchanged unless a link to HUMANS becomes stale.
- Link to `AGENTS.md` for judge-mode thresholds and design invariants.
- Include profiles for smoke tests, small pairwise runs, larger scored runs, low-memory MoE, and report comparison.
- Run doc governance mentally against the three-tier split before committing.

## Test Plan

- `uv run pre-commit run --all-files`
- Manual doc governance check: README stays brief, HUMANS owns usage, AGENTS owns invariants.
