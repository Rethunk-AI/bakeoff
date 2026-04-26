# Tasks

## P0

- [ ] Define the comparison input contract for two result JSON files.
- [ ] Reuse existing per-model rollup math for comparison.
- [ ] Compute deltas for latency mean, tokens/sec, energy, cost, and heuristic quality.
- [ ] Compare scored judge means when both runs contain scored judgements.
- [ ] Compare pairwise win rates when both runs contain pairwise judgements.
- [ ] Emit compatibility warnings for changed tasks, prompts, model IDs, judge mode, seed, or config hash.
- [ ] Add Markdown comparison output.
- [ ] Add a CLI entry point and help text.
- [ ] Add unit tests with compact result fixtures.

## P1

- [ ] Add output-to-file support.
- [ ] Add strict mode that exits non-zero on compatibility warnings.
- [ ] Document examples for prompt changes, model changes, and quantization changes.

## P2

- [ ] Consider an HTML comparison dashboard after Markdown output stabilizes.
- [ ] Consider multi-run trend input after the two-run path is proven.
