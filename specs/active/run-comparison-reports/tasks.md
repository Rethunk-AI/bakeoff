# Tasks

## P0

- [x] Define the comparison input contract for two result JSON files.
- [x] Reuse existing per-model rollup math for comparison.
- [x] Compute deltas for latency mean, tokens/sec, energy, cost, and heuristic quality.
- [x] Compare scored judge means when both runs contain scored judgements.
- [x] Compare pairwise win rates when both runs contain pairwise judgements.
- [x] Emit compatibility warnings for changed tasks, prompts, model IDs, judge mode, seed, or config hash.
- [x] Add Markdown comparison output.
- [x] Add a CLI entry point and help text.
- [x] Add unit tests with compact result fixtures.

## P1

- [x] Add output-to-file support.
- [x] Add strict mode that exits non-zero on compatibility warnings.
- [x] Document examples for prompt changes, model changes, and quantization changes.

## P2

- [ ] Consider an HTML comparison dashboard after Markdown output stabilizes.
- [ ] Consider multi-run trend input after the two-run path is proven.
