# Benchmark Resume And Partial Rerun

Status: Active

## Problem

Long benchmark runs can fail after some model phases have completed. Today the operator must rerun the whole matrix, which repeats expensive model swaps, energy measurements, and judge calls. There is no supported way to resume from an existing result payload or rerun only failed cells.

## Goals

Allow an interrupted benchmark to continue from an existing JSON result while preserving per-model sequential execution. Support targeted reruns for missing or errored model rows and stale judge rows. Make resumed output explicit so downstream reports can distinguish reused records from freshly measured records.

## Non-Goals

This spec does not add concurrent execution or round-robin scheduling. It does not merge unrelated configs automatically. It does not try to compare semantically changed prompts or datasets beyond deterministic identity checks.

## Design

Add resume inputs to the runner, likely `--resume-from results/run-*.json` plus focused selectors such as `--rerun-errors`, `--rerun-missing`, and optional model/task/prompt filters. The runner should load the prior payload, validate that the current config, seed, generated tasks, prompt IDs, and model IDs are compatible, then build a pending matrix.

Execution must remain per-model sequential: for each model with pending cells, warm up once and run all pending task/prompt cells for that model. Reused rows should not trigger model swaps. Fresh rows should include metadata such as `source_run_id`, `resumed_from`, or `attempt` so reports can explain mixed provenance.

Judge resume should be separate and derived from available model records. Pairwise mode should only judge missing pair/task/prompt records. Scored mode should only judge missing model/task/prompt records. Operators should be able to skip judge resume when they only want model rows.

The output should be a new result file, not an in-place mutation of the old run, to preserve auditability.

## Acceptance Criteria

- Resume creates a new JSON/Markdown/HTML result file and leaves the source run unchanged.
- Compatible prior rows are reused without re-running their model cells.
- Missing or errored rows can be rerun by model/task/prompt key while preserving per-model sequential execution.
- Judge resume handles pairwise and scored modes without duplicating existing judgements.
- Incompatible resume inputs fail early with clear reasons.
- Tests prove phase ordering for normal runs and resumed partial runs.
