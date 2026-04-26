# Benchmark Profile Documentation

Status: Active

## Problem

Operators can run the benchmark, but the documentation does not yet provide recommended profiles for common goals. This makes it easy to choose an unnecessarily expensive judge mode, dataset size, model set, or MoE setting.

## Goals

Document practical benchmark profiles that map operator intent to safe config choices. Keep the guidance in `HUMANS.md`, because it is operator-facing usage documentation. Cross-reference design invariants in `AGENTS.md` without duplicating them.

## Non-Goals

This spec does not add profile presets to the CLI. It does not create a new docs directory or duplicate install instructions in README. It does not change benchmark defaults unless another implementation spec justifies that change.

## Design

Add a concise `Recommended benchmark profiles` section to `HUMANS.md`. Include profiles for quick smoke, two-model serious run, three-to-four-model pairwise tournament, five-or-more-model scored run, low-memory MoE run, and report-only comparison workflow once comparison exists.

Each profile should describe when to use it, config fields to edit, expected cost shape, and what output to inspect. The section should link to existing judge mode selection guidance in `AGENTS.md` and troubleshooting entries in `HUMANS.md`.

If later implementation adds reusable sample configs, this spec can be extended to cover `examples/` or `config/*.yaml`; for now, keep it documentation-only.

## Acceptance Criteria

- `HUMANS.md` contains operator-facing benchmark profile guidance.
- README remains compact and does not duplicate the new usage detail.
- AGENTS keeps design invariants and does not become an operator guide.
- Profiles mention judge mode cost implications and the one-model-in-VRAM invariant by link, not by duplicating long explanations.
- Documentation passes pre-commit.
