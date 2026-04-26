# Run Provenance And Model Metadata

Status: Active

## Problem

Benchmark results currently include timestamp, config, tasks, records, and judgements, but they do not capture enough context to reproduce or interpret a run later. Missing context includes git SHA, config hash, host/runtime details, dependency versions, llama-swap version, container image, model capability fields, and upstream HuggingFace metadata.

## Goals

Add reproducible run provenance to JSON, Markdown, and HTML outputs. Add structured model capability metadata to the result payload. Enrich model metadata from HuggingFace when it is cheap and available, while keeping normal benchmark runs deterministic and usable offline.

## Non-Goals

This spec does not require network access for benchmark execution. It does not download models, verify file checksums for large GGUFs, or guarantee that HuggingFace metadata is complete. Model download behavior remains owned by `bench.download`.

## Design

Introduce a `bench.provenance` module that collects local run context with bounded subprocess calls and best-effort fallbacks. The payload should gain a top-level `provenance` object containing git, config, runtime, host, dependency, proxy, and output metadata. Missing tools should produce `null` fields plus a warning list, not hard failures.

Model entries should gain a normalized `model_metadata` section in the result payload. Local fields come from `config.yaml`: `id`, `alias`, `gguf`, `ctx`, `n_cpu_moe`, server defaults that affect serving, and inferred fields such as quantization and repo ID when safe. Optional HuggingFace enrichment should look up repo/file metadata for `<org>/<repo>/<file>.gguf` entries and record fields such as repo, filename, size, last modified, commit revision, and URL. The enrichment path should be opt-in by config or CLI so offline benchmark runs remain predictable.

Markdown and HTML report headers should surface the most useful provenance fields: git SHA, config hash, seed, Python version, platform, llama-swap version if known, container image, and model metadata summary. JSON remains the source of truth.

## Acceptance Criteria

- New runs include a `provenance` object and `model_metadata` list in JSON.
- Markdown and HTML reports expose concise provenance headers without overwhelming the rollup.
- Missing git, GPU tools, HuggingFace credentials, or network access do not fail a benchmark unless the user requested strict enrichment.
- HuggingFace API calls are optional, bounded, and covered by tests with mocks.
- Existing report readers remain compatible with older payloads that lack provenance.
