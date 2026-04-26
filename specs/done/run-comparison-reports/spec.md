# Run Comparison Reports

Status: Active

## Problem

The report layer summarizes one benchmark run at a time. Operators who change prompts, models, quantization, context length, or infrastructure need a supported way to compare two or more result files across latency, throughput, quality, judge results, energy, and cost.

## Goals

Add a run comparison command that reads existing result JSON files and emits a clear comparison summary. Support the common two-run case first, while leaving room for multi-run trend views. Reuse report rollup logic where possible so single-run and comparison math stay consistent.

## Non-Goals

This spec does not require a persistent database, hosted dashboard, or automatic benchmark execution. It does not try to compare incompatible datasets without warning.

## Design

Add comparison helpers to `bench.report` or a focused `bench.compare` module. The comparison should load result JSON files, compute per-model rollups using the same functions as normal reports, and align rows by model ID. The output should include deltas for latency, tokens/sec, heuristic quality, judge score or win rate, energy, and cost where available.

Expose the feature through a CLI path such as `python -m bench.report compare <base.json> <candidate.json>` or `python -m bench.compare <base.json> <candidate.json>`. Markdown output should be the first deliverable because it is scriptable and easy to review. HTML comparison can follow if the data model is stable.

Compatibility checks should warn when runs have different task sets, prompt IDs, judge modes, seeds, or config hashes. Warnings should not block comparison unless the operator requests strict mode.

## Acceptance Criteria

- A documented CLI compares two result JSON files and emits Markdown to stdout or a file.
- Comparison rows include absolute values and deltas for core metrics.
- Judge comparison works for pairwise and scored modes when data exists.
- Incompatible runs produce visible warnings.
- Tests cover metric deltas, missing metrics, judge modes, and compatibility warnings.
- Existing single-run reports continue to render unchanged.
