# Config Validation And Dry-Run Fixture

Status: Active

## Problem

`config.yaml` is the benchmark contract, but validation is currently split across the runner, the llama-swap generator, and runtime failures. Some invalid inputs are only caught when a live benchmark starts. The dry-run path exercises dataset generation and llama-swap config rendering, but it does not provide a structured validation report or fixture coverage for the major config branches.

## Goals

Add a first-class config validation layer that reports actionable errors before proxy startup. Keep validation local, deterministic, and independent of model downloads or GPU availability. Expand dry-run coverage so CI can catch invalid benchmark shapes without running inference.

## Non-Goals

This spec does not add schema-generation tooling, migrate `config.yaml` to another format, or validate that GGUF files exist on disk. File existence and HuggingFace metadata checks belong to the provenance and metadata spec.

## Design

Introduce a small `bench.config` module with `ConfigError`, `ValidationIssue`, `load_config`, and `validate_config` helpers. The runner should call this module immediately after loading YAML, before dataset generation or llama-swap config generation. The llama-swap generator can keep defensive checks, but user-facing validation errors should point to config paths such as `models[1].gguf`.

Validation should cover required sections, duplicate IDs, valid judge modes, positive numeric fields, non-empty `models` and `prompts`, `mmproj-*` rejection, prompt IDs, output settings, and dataset domains. Errors should be aggregated when practical so operators can fix multiple config issues in one edit.

Dry-run should emit a concise validation summary and continue to exercise dataset generation, llama-swap config rendering, and YAML serialization. Add tests for both valid and invalid fixture configs.

## Acceptance Criteria

- Invalid config exits before proxy startup with clear path-oriented errors.
- Dry-run still succeeds for the repository `config.yaml`.
- Tests cover duplicate model IDs, bad judge mode, empty prompts, invalid numeric values, `mmproj-*` entries, and valid minimal config.
- Existing llama-swap structural tests continue to pass.
- Documentation mentions validation behavior where operators already learn about dry-run.
